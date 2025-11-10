# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Access datasets."""
import filecmp
import importlib
import inspect
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union
from urllib.parse import urlparse

from datasets import DownloadConfig, DownloadMode
from datasets.builder import DatasetBuilder
from datasets.packaged_modules import _EXTENSION_TO_MODULE, _hash_python_lines
from datasets.utils.filelock import FileLock
from datasets.utils.version import Version

from . import SCRIPTS_VERSION, config
from .module import EvaluationModule
from .utils.file_utils import (
    cached_path,
    head_hf_s3,
    hf_hub_url,
    init_hf_modules,
    is_relative_path,
    relative_to_absolute_path,
    url_or_path_join,
)
from .utils.logging import get_logger


logger = get_logger(__name__)


ALL_ALLOWED_EXTENSIONS = list(_EXTENSION_TO_MODULE.keys()) + ["zip"]


def init_dynamic_modules(
    name: str = config.MODULE_NAME_FOR_DYNAMIC_MODULES, hf_modules_cache: Optional[Union[Path, str]] = None
):
    """
    Create a module with name `name` in which you can add dynamic modules
    such as metrics or datasets. The module can be imported using its name.
    The module is created in the HF_MODULE_CACHE directory by default (~/.cache/huggingface/modules) but it can
    be overriden by specifying a path to another directory in `hf_modules_cache`.
    """
    hf_modules_cache = init_hf_modules(hf_modules_cache)
    dynamic_modules_path = os.path.join(hf_modules_cache, name)
    os.makedirs(dynamic_modules_path, exist_ok=True)
    if not os.path.exists(os.path.join(dynamic_modules_path, "__init__.py")):
        with open(os.path.join(dynamic_modules_path, "__init__.py"), "w"):
            pass
    return dynamic_modules_path


def import_main_class(module_path) -> Optional[Union[Type[DatasetBuilder], Type[EvaluationModule]]]:
    """Import a module at module_path and return its main class, a Metric by default"""
    module = importlib.import_module(module_path)
    main_cls_type = EvaluationModule

    # Find the main class in our imported module
    module_main_cls = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, main_cls_type):
            if inspect.isabstract(obj):
                continue
            module_main_cls = obj
            break

    return module_main_cls


def files_to_hash(file_paths: List[str]) -> str:
    """
    Convert a list of scripts or text files provided in file_paths into a hashed filename in a repeatable way.
    """
    # List all python files in directories if directories are supplied as part of external imports
    to_use_files: List[Union[Path, str]] = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            to_use_files.extend(list(Path(file_path).rglob("*.[pP][yY]")))
        else:
            to_use_files.append(file_path)

    # Get the code from all these files
    lines = []
    for file_path in to_use_files:
        with open(file_path, encoding="utf-8") as f:
            lines.extend(f.readlines())
    return _hash_python_lines(lines)


def convert_github_url(url_path: str) -> Tuple[str, Optional[str]]:
    """Convert a link to a file on a github repo in a link to the raw github object."""
    parsed = urlparse(url_path)
    sub_directory = None
    if parsed.scheme in ("http", "https", "s3") and parsed.netloc == "github.com":
        if "blob" in url_path:
            if not url_path.endswith(".py"):
                raise ValueError(f"External import from github at {url_path} should point to a file ending with '.py'")
            url_path = url_path.replace("blob", "raw")  # Point to the raw file
        else:
            # Parse github url to point to zip
            github_path = parsed.path[1:]
            repo_info, branch = github_path.split("/tree/") if "/tree/" in github_path else (github_path, "master")
            repo_owner, repo_name = repo_info.split("/")
            url_path = f"https://github.com/{repo_owner}/{repo_name}/archive/{branch}.zip"
            sub_directory = f"{repo_name}-{branch}"
    return url_path, sub_directory


def increase_load_count(name: str, resource_type: str):
    """Update the download count of a dataset or metric."""
    if not config.HF_EVALUATE_OFFLINE and config.HF_UPDATE_DOWNLOAD_COUNTS:
        try:
            head_hf_s3(name, filename=name + ".py", dataset=(resource_type == "dataset"))
        except Exception:
            pass


def get_imports(file_path: str) -> Tuple[str, str, str, str]:
    """Find whether we should import or clone additional files for a given processing script.
        And list the import.

    We allow:
    - library dependencies,
    - local dependencies and
    - external dependencies whose url is specified with a comment starting from "# From:' followed by the raw url to a file, an archive or a github repository.
        external dependencies will be downloaded (and extracted if needed in the dataset folder).
        We also add an `__init__.py` to each sub-folder of a downloaded folder so the user can import from them in the script.

    Note that only direct import in the dataset processing script will be handled
    We don't recursively explore the additional import to download further files.

    Example::

        import tensorflow
        import .c4_utils
        import .clicr.dataset-code.build_json_dataset  # From: https://raw.githubusercontent.com/clips/clicr/master/dataset-code/build_json_dataset
    """
    lines = []
    with open(file_path, encoding="utf-8") as f:
        lines.extend(f.readlines())

    logger.debug(f"Checking {file_path} for additional imports.")
    imports: List[Tuple[str, str, str, Optional[str]]] = []
    is_in_docstring = False
    for line in lines:
        docstr_start_match = re.findall(r'[\s\S]*?"""[\s\S]*?', line)

        if len(docstr_start_match) == 1:
            # flip True <=> False only if doctstring
            # starts at line without finishing
            is_in_docstring = not is_in_docstring

        if is_in_docstring:
            # import statements in doctstrings should
            # not be added as required dependencies
            continue

        match = re.match(r"^import\s+(\.?)([^\s\.]+)[^#\r\n]*(?:#\s+From:\s+)?([^\r\n]*)", line, flags=re.MULTILINE)
        if match is None:
            match = re.match(
                r"^from\s+(\.?)([^\s\.]+)(?:[^\s]*)\s+import\s+[^#\r\n]*(?:#\s+From:\s+)?([^\r\n]*)",
                line,
                flags=re.MULTILINE,
            )
            if match is None:
                continue
        if match.group(1):
            # The import starts with a '.', we will download the relevant file
            if any(imp[1] == match.group(2) for imp in imports):
                # We already have this import
                continue
            if match.group(3):
                # The import has a comment with 'From:', we'll retrieve it from the given url
                url_path = match.group(3)
                url_path, sub_directory = convert_github_url(url_path)
                imports.append(("external", match.group(2), url_path, sub_directory))
            elif match.group(2):
                # The import should be at the same place as the file
                imports.append(("internal", match.group(2), match.group(2), None))
        else:
            if match.group(3):
                # The import has a comment with `From: git+https:...`, asks user to pip install from git.
                url_path = match.group(3)
                imports.append(("library", match.group(2), url_path, None))
            else:
                imports.append(("library", match.group(2), match.group(2), None))

    return imports


def _download_additional_modules(
    name: str, base_path: str, imports: Tuple[str, str, str, str], download_config: Optional[DownloadConfig]
) -> List[Tuple[str, str]]:
    """
    Download additional module for a module <name>.py at URL (or local path) <base_path>/<name>.py
    The imports must have been parsed first using ``get_imports``.

    If some modules need to be installed with pip, an error is raised showing how to install them.
    This function return the list of downloaded modules as tuples (import_name, module_file_path).

    The downloaded modules can then be moved into an importable directory with ``_copy_script_and_other_resources_in_importable_dir``.
    """
    local_imports = []
    library_imports = []
    download_config = download_config.copy()
    if download_config.download_desc is None:
        download_config.download_desc = "Downloading extra modules"
    for import_type, import_name, import_path, sub_directory in imports:
        if import_type == "library":
            library_imports.append((import_name, import_path))  # Import from a library
            continue

        if import_name == name:
            raise ValueError(
                f"Error in the {name} script, importing relative {import_name} module "
                f"but {import_name} is the name of the script. "
                f"Please change relative import {import_name} to another name and add a '# From: URL_OR_PATH' "
                f"comment pointing to the original relative import file path."
            )
        if import_type == "internal":
            url_or_filename = url_or_path_join(base_path, import_path + ".py")
        elif import_type == "external":
            url_or_filename = import_path
        else:
            raise ValueError("Wrong import_type")

        local_import_path = cached_path(
            url_or_filename,
            download_config=download_config,
        )
        if sub_directory is not None:
            local_import_path = os.path.join(local_import_path, sub_directory)
        local_imports.append((import_name, local_import_path))

    # Check library imports
    needs_to_be_installed = set()
    for library_import_name, library_import_path in library_imports:
        try:
            lib = importlib.import_module(library_import_name)  # noqa F841
        except ImportError:
            library_import_name = "scikit-learn" if library_import_name == "sklearn" else library_import_name
            needs_to_be_installed.add((library_import_name, library_import_path))
    if needs_to_be_installed:
        raise ImportError(
            f"To be able to use {name}, you need to install the following dependencies"
            f"{[lib_name for lib_name, lib_path in needs_to_be_installed]} using 'pip install "
            f"{' '.join([lib_name for lib_name, lib_path in needs_to_be_installed])}' for instance'"
        )
    return local_imports


def _copy_script_and_other_resources_in_importable_dir(
    name: str,
    importable_directory_path: str,
    subdirectory_name: str,
    original_local_path: str,
    local_imports: List[Tuple[str, str]],
    additional_files: List[Tuple[str, str]],
    download_mode: Optional[DownloadMode],
) -> str:
    """Copy a script and its required imports to an importable directory

    Args:
        name (str): name of the resource to load
        importable_directory_path (str): path to the loadable folder in the dynamic modules directory
        subdirectory_name (str): name of the subdirectory in importable_directory_path in which to place the script
        original_local_path (str): local path to the resource script
        local_imports (List[Tuple[str, str]]): list of (destination_filename, import_file_to_copy)
        additional_files (List[Tuple[str, str]]): list of (destination_filename, additional_file_to_copy)
        download_mode (Optional[DownloadMode]): download mode

    Return:
        importable_local_file: path to an importable module with importlib.import_module
    """

    # Define a directory with a unique name in our dataset or metric folder
    # path is: ./datasets|metrics/dataset|metric_name/hash_from_code/script.py
    # we use a hash as subdirectory_name to be able to have multiple versions of a dataset/metric processing file together
    importable_subdirectory = os.path.join(importable_directory_path, subdirectory_name)
    importable_local_file = os.path.join(importable_subdirectory, name + ".py")

    # Prevent parallel disk operations
    lock_path = importable_directory_path + ".lock"
    with FileLock(lock_path):
        # Create main dataset/metrics folder if needed
        if download_mode == DownloadMode.FORCE_REDOWNLOAD and os.path.exists(importable_directory_path):
            shutil.rmtree(importable_directory_path)
        os.makedirs(importable_directory_path, exist_ok=True)

        # add an __init__ file to the main dataset folder if needed
        init_file_path = os.path.join(importable_directory_path, "__init__.py")
        if not os.path.exists(init_file_path):
            with open(init_file_path, "w"):
                pass

        # Create hash dataset folder if needed
        os.makedirs(importable_subdirectory, exist_ok=True)
        # add an __init__ file to the hash dataset folder if needed
        init_file_path = os.path.join(importable_subdirectory, "__init__.py")
        if not os.path.exists(init_file_path):
            with open(init_file_path, "w"):
                pass

        # Copy dataset.py file in hash folder if needed
        if not os.path.exists(importable_local_file):
            shutil.copyfile(original_local_path, importable_local_file)

        # Record metadata associating original dataset path with local unique folder
        meta_path = importable_local_file.split(".py")[0] + ".json"
        if not os.path.exists(meta_path):
            meta = {"original file path": original_local_path, "local file path": importable_local_file}
            # the filename is *.py in our case, so better rename to filenam.json instead of filename.py.json
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

        # Copy all the additional imports
        for import_name, import_path in local_imports:
            if os.path.isfile(import_path):
                full_path_local_import = os.path.join(importable_subdirectory, import_name + ".py")
                if not os.path.exists(full_path_local_import):
                    shutil.copyfile(import_path, full_path_local_import)
            elif os.path.isdir(import_path):
                full_path_local_import = os.path.join(importable_subdirectory, import_name)
                if not os.path.exists(full_path_local_import):
                    shutil.copytree(import_path, full_path_local_import)
            else:
                raise OSError(f"Error with local import at {import_path}")

        # Copy aditional files like dataset infos file if needed
        for file_name, original_path in additional_files:
            destination_additional_path = os.path.join(importable_subdirectory, file_name)
            if not os.path.exists(destination_additional_path) or not filecmp.cmp(
                original_path, destination_additional_path
            ):
                shutil.copyfile(original_path, destination_additional_path)
        return importable_local_file


def _create_importable_file(
    local_path: str,
    local_imports: List[Tuple[str, str]],
    additional_files: List[Tuple[str, str]],
    dynamic_modules_path: str,
    module_namespace: str,
    name: str,
    download_mode: DownloadMode,
) -> Tuple[str, str]:
    importable_directory_path = os.path.join(dynamic_modules_path, module_namespace, name.replace("/", "--"))
    Path(importable_directory_path).mkdir(parents=True, exist_ok=True)
    (Path(importable_directory_path).parent / "__init__.py").touch(exist_ok=True)
    hash = files_to_hash([local_path] + [loc[1] for loc in local_imports])
    importable_local_file = _copy_script_and_other_resources_in_importable_dir(
        name=name.split("/")[-1],
        importable_directory_path=importable_directory_path,
        subdirectory_name=hash,
        original_local_path=local_path,
        local_imports=local_imports,
        additional_files=additional_files,
        download_mode=download_mode,
    )
    logger.debug(f"Created importable dataset file at {importable_local_file}")
    module_path = ".".join(
        [os.path.basename(dynamic_modules_path), module_namespace, name.replace("/", "--"), hash, name.split("/")[-1]]
    )
    return module_path, hash


@dataclass
class ImportableModule:
    module_path: str
    hash: str


class _EvaluationModuleFactory:
    def get_module(self) -> ImportableModule:
        raise NotImplementedError


class LocalEvaluationModuleFactory(_EvaluationModuleFactory):
    """Get the module of a local metric. The metric script is loaded from a local script."""

    def __init__(
        self,
        path: str,
        module_type: str = "metrics",
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[DownloadMode] = None,
        dynamic_modules_path: Optional[str] = None,
    ):
        self.path = path
        self.module_type = module_type
        self.name = Path(path).stem
        self.download_config = download_config or DownloadConfig()
        self.download_mode = download_mode
        self.dynamic_modules_path = dynamic_modules_path

    def get_module(self) -> ImportableModule:
        # get script and other files
        imports = get_imports(self.path)
        local_imports = _download_additional_modules(
            name=self.name,
            base_path=str(Path(self.path).parent),
            imports=imports,
            download_config=self.download_config,
        )
        # copy the script and the files in an importable directory
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        module_path, hash = _create_importable_file(
            local_path=self.path,
            local_imports=local_imports,
            additional_files=[],
            dynamic_modules_path=dynamic_modules_path,
            module_namespace=self.module_type,
            name=self.name,
            download_mode=self.download_mode,
        )
        # make the new module to be noticed by the import system
        importlib.invalidate_caches()
        return ImportableModule(module_path, hash)


class HubEvaluationModuleFactory(_EvaluationModuleFactory):
    """Get the module of a metric from a metric repository on the Hub."""

    def __init__(
        self,
        name: str,
        module_type: str = "metrics",
        revision: Optional[Union[str, Version]] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[DownloadMode] = None,
        dynamic_modules_path: Optional[str] = None,
    ):
        self.name = name
        self.module_type = module_type
        self.revision = revision
        self.download_config = download_config or DownloadConfig()
        self.download_mode = download_mode
        self.dynamic_modules_path = dynamic_modules_path
        assert self.name.count("/") == 1
        increase_load_count(name, resource_type="metric")

    def download_loading_script(self, revision) -> str:
        file_path = hf_hub_url(path=self.name, name=self.name.split("/")[1] + ".py", revision=revision)
        download_config = self.download_config.copy()
        if download_config.download_desc is None:
            download_config.download_desc = "Downloading builder script"
        return cached_path(file_path, download_config=download_config)

    def get_module(self) -> ImportableModule:
        revision = self.revision or os.getenv("HF_SCRIPTS_VERSION", SCRIPTS_VERSION)

        if re.match(r"\d*\.\d*\.\d*", revision):  # revision is version number (three digits separated by full stops)
            revision = "v" + revision  # tagging convention on evaluate repository starts with v

        # get script and other files
        try:
            local_path = self.download_loading_script(revision)
        except FileNotFoundError as err:
            # if there is no file found with current revision tag try to load main
            if self.revision is None and os.getenv("HF_SCRIPTS_VERSION", SCRIPTS_VERSION) != "main":
                revision = "main"
                local_path = self.download_loading_script(revision)
            else:
                raise err

        imports = get_imports(local_path)
        local_imports = _download_additional_modules(
            name=self.name,
            base_path=hf_hub_url(path=self.name, name="", revision=revision),
            imports=imports,
            download_config=self.download_config,
        )
        # copy the script and the files in an importable directory
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        module_path, hash = _create_importable_file(
            local_path=local_path,
            local_imports=local_imports,
            additional_files=[],
            dynamic_modules_path=dynamic_modules_path,
            module_namespace=self.module_type,
            name=self.name,
            download_mode=self.download_mode,
        )
        # make the new module to be noticed by the import system
        importlib.invalidate_caches()
        return ImportableModule(module_path, hash)


class CachedEvaluationModuleFactory(_EvaluationModuleFactory):
    """
    Get the module of a metric that has been loaded once already and cached.
    The script that is loaded from the cache is the most recent one with a matching name.
    """

    def __init__(
        self,
        name: str,
        module_type: str = "metrics",
        dynamic_modules_path: Optional[str] = None,
    ):
        self.name = name
        self.module_type = module_type
        self.dynamic_modules_path = dynamic_modules_path
        assert self.name.count("/") == 0

    def get_module(self) -> ImportableModule:
        dynamic_modules_path = self.dynamic_modules_path if self.dynamic_modules_path else init_dynamic_modules()
        importable_directory_path = os.path.join(dynamic_modules_path, self.module_type, self.name)
        hashes = (
            [h for h in os.listdir(importable_directory_path) if len(h) == 64]
            if os.path.isdir(importable_directory_path)
            else None
        )
        if not hashes:
            raise FileNotFoundError(f"Metric {self.name} is not cached in {dynamic_modules_path}")
        # get most recent

        def _get_modification_time(module_hash):
            return (
                (Path(importable_directory_path) / module_hash / (self.name.split("--")[-1] + ".py")).stat().st_mtime
            )

        hash = sorted(hashes, key=_get_modification_time)[-1]
        logger.warning(
            f"Using the latest cached version of the module from {os.path.join(importable_directory_path, hash)} "
            f"(last modified on {time.ctime(_get_modification_time(hash))}) since it "
            f"couldn't be found locally at {self.name}, or remotely on the Hugging Face Hub."
        )
        # make the new module to be noticed by the import system
        module_path = ".".join(
            [os.path.basename(dynamic_modules_path), self.module_type, self.name, hash, self.name.split("--")[-1]]
        )
        importlib.invalidate_caches()
        return ImportableModule(module_path, hash)


def evaluation_module_factory(
    path: str,
    module_type: Optional[str] = None,
    revision: Optional[Union[str, Version]] = None,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[DownloadMode] = None,
    force_local_path: Optional[str] = None,
    dynamic_modules_path: Optional[str] = None,
    **download_kwargs,
) -> ImportableModule:
    """
    Download/extract/cache a metric module.

    Metrics codes are cached inside the the dynamic modules cache to allow easy import (avoid ugly sys.path tweaks).

    Args:

        path (str): Path or name of the metric script.

            - if ``path`` is a local metric script or a directory containing a local metric script (if the script has the same name as the directory):
              -> load the module from the metric script
              e.g. ``'./metrics/accuracy'`` or ``'./metrics/accuracy/accuracy.py'``.
            - if ``path`` is a metric on the Hugging Face Hub (ex: `glue`, `squad`)
              -> load the module from the metric script in the github repository at huggingface/datasets
              e.g. ``'accuracy'`` or ``'rouge'``.

        revision (Optional ``Union[str, datasets.Version]``):
            If specified, the module will be loaded from the datasets repository at this version.
            By default:
            - it is set to the local version of the lib.
            - it will also try to load it from the master branch if it's not available at the local version of the lib.
            Specifying a version that is different from your local version of the lib might cause compatibility issues.
        download_config (:class:`DownloadConfig`, optional): Specific download configuration parameters.
        download_mode (:class:`DownloadMode`, default ``REUSE_DATASET_IF_EXISTS``): Download/generate mode.
        force_local_path (Optional str): Optional path to a local path to download and prepare the script to.
            Used to inspect or modify the script folder.
        dynamic_modules_path (Optional str, defaults to HF_MODULES_CACHE / "datasets_modules", i.e. ~/.cache/huggingface/modules/datasets_modules):
            Optional path to the directory in which the dynamic modules are saved. It must have been initialized with :obj:`init_dynamic_modules`.
            By default the datasets and metrics are stored inside the `datasets_modules` module.
        download_kwargs: optional attributes for DownloadConfig() which will override the attributes in download_config if supplied.

    Returns:
        ImportableModule
    """
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    download_config.extract_compressed_file = True
    download_config.force_extract = True

    filename = list(filter(lambda x: x, path.replace(os.sep, "/").split("/")))[-1]
    if not filename.endswith(".py"):
        filename = filename + ".py"
    combined_path = os.path.join(path, filename)
    # Try locally
    if path.endswith(filename):
        if os.path.isfile(path):
            return LocalEvaluationModuleFactory(
                path, download_mode=download_mode, dynamic_modules_path=dynamic_modules_path
            ).get_module()
        else:
            raise FileNotFoundError(f"Couldn't find a metric script at {relative_to_absolute_path(path)}")
    elif os.path.isfile(combined_path):
        return LocalEvaluationModuleFactory(
            combined_path, download_mode=download_mode, dynamic_modules_path=dynamic_modules_path
        ).get_module()
    elif is_relative_path(path) and path.count("/") <= 1 and not force_local_path:
        try:
            # load a canonical evaluation module from hub
            if path.count("/") == 0:
                # if no type provided look through all possible modules
                if module_type is None:
                    for current_type in ["metric", "comparison", "measurement"]:
                        try:
                            return HubEvaluationModuleFactory(
                                f"evaluate-{current_type}/{path}",
                                revision=revision,
                                download_config=download_config,
                                download_mode=download_mode,
                                dynamic_modules_path=dynamic_modules_path,
                            ).get_module()
                        except ConnectionError:
                            pass
                    raise FileNotFoundError
                # if module_type provided load specific module_type
                else:
                    return HubEvaluationModuleFactory(
                        f"evaluate-{module_type}/{path}",
                        revision=revision,
                        download_config=download_config,
                        download_mode=download_mode,
                        dynamic_modules_path=dynamic_modules_path,
                    ).get_module()
            # load community evaluation module from hub
            elif path.count("/") == 1:
                return HubEvaluationModuleFactory(
                    path,
                    revision=revision,
                    download_config=download_config,
                    download_mode=download_mode,
                    dynamic_modules_path=dynamic_modules_path,
                ).get_module()
        except Exception as e1:  # noqa: all the attempts failed, before raising the error we should check if the module is already cached.
            # if it's a canonical module we need to check if it's any of the types
            if path.count("/") == 0:
                for current_type in ["metric", "comparison", "measurement"]:
                    try:
                        return CachedEvaluationModuleFactory(
                            f"evaluate-{current_type}--{path}", dynamic_modules_path=dynamic_modules_path
                        ).get_module()
                    except Exception as e2:  # noqa: if it's not in the cache, then it doesn't exist.
                        pass
            # if it's a community module we just need to check on path
            elif path.count("/") == 1:
                try:
                    return CachedEvaluationModuleFactory(
                        path.replace("/", "--"), dynamic_modules_path=dynamic_modules_path
                    ).get_module()
                except Exception as e2:  # noqa: if it's not in the cache, then it doesn't exist.
                    pass
            if not isinstance(e1, (ConnectionError, FileNotFoundError)):
                raise e1 from None
            raise FileNotFoundError(
                f"Couldn't find a module script at {relative_to_absolute_path(combined_path)}. "
                f"Module '{path}' doesn't exist on the Hugging Face Hub either."
            ) from None
    else:
        raise FileNotFoundError(f"Couldn't find a module script at {relative_to_absolute_path(combined_path)}.")


def load(
    path: str,
    config_name: Optional[str] = None,
    module_type: Optional[str] = None,
    process_id: int = 0,
    num_process: int = 1,
    cache_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
    keep_in_memory: bool = False,
    download_config: Optional[DownloadConfig] = None,
    download_mode: Optional[DownloadMode] = None,
    revision: Optional[Union[str, Version]] = None,
    **init_kwargs,
) -> EvaluationModule:
    """Load a [`~evaluate.EvaluationModule`].

    Args:

        path (`str`):
            Path to the evaluation processing script with the evaluation builder. Can be either:
                - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
                    e.g. `'./metrics/rouge'` or `'./metrics/rouge/rouge.py'`
                - a evaluation module identifier on the HuggingFace evaluate repo e.g. `'rouge'` or `'bleu'` that are in either `'metrics/'`,
                    `'comparisons/'`, or `'measurements/'` depending on the provided `module_type`
        config_name (`str`, *optional*):
            Selecting a configuration for the metric (e.g. the GLUE metric has a configuration for each subset).
        module_type (`str`, default `'metric'`):
            Type of evaluation module, can be one of `'metric'`, `'comparison'`, or `'measurement'`.
        process_id (`int`, *optional*):
            For distributed evaluation: id of the process.
        num_process (`int`, *optional*):
            For distributed evaluation: total number of processes.
        cache_dir (`str`, *optional*):
            Path to store the temporary predictions and references (default to `~/.cache/huggingface/evaluate/`).
        experiment_id (`str`):
            A specific experiment id. This is used if several distributed evaluations share the same file system.
            This is useful to compute metrics in distributed setups (in particular non-additive metrics like F1).
        keep_in_memory (`bool`):
            Whether to store the temporary results in memory (defaults to `False`).
        download_config ([`~evaluate.DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`], defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        revision (`Union[str, evaluate.Version]`, *optional*):
            If specified, the module will be loaded from the datasets repository
            at this version. By default it is set to the local version of the lib. Specifying a version that is different from
            your local version of the lib might cause compatibility issues.

    Returns:
        [`evaluate.EvaluationModule`]

    Example:

        ```py
        >>> from evaluate import load
        >>> accuracy = load("accuracy")
        ```
    """
    download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
    evaluation_module = evaluation_module_factory(
        path, module_type=module_type, revision=revision, download_config=download_config, download_mode=download_mode
    )
    evaluation_cls = import_main_class(evaluation_module.module_path)
    evaluation_instance = evaluation_cls(
        config_name=config_name,
        process_id=process_id,
        num_process=num_process,
        cache_dir=cache_dir,
        keep_in_memory=keep_in_memory,
        experiment_id=experiment_id,
        hash=evaluation_module.hash,
        **init_kwargs,
    )

    if module_type and module_type != evaluation_instance.module_type:
        raise TypeError(
            f"No module of module type '{module_type}' not found for '{path}' locally, or on the Hugging Face Hub. Found module of module type '{evaluation_instance.module_type}' instead."
        )

    # Download and prepare resources for the metric
    evaluation_instance.download_and_prepare(download_config=download_config)

    return evaluation_instance

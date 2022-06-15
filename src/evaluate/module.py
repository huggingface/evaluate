# Copyright 2020 The HuggingFace Datasets Authors
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
""" EvaluationModule base class."""
import os
import types
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
from datasets import DatasetInfo
from datasets.arrow_dataset import Dataset
from datasets.arrow_reader import ArrowReader
from datasets.arrow_writer import ArrowWriter
from datasets.features import Features, Sequence, Value
from datasets.features.features import _check_non_null_non_empty_recursive
from datasets.utils.download_manager import DownloadManager
from datasets.utils.filelock import BaseFileLock, FileLock, Timeout
from datasets.utils.py_utils import copyfunc, temp_seed, zip_dict

from . import config
from .info import EvaluationModuleInfo
from .naming import camelcase_to_snakecase
from .utils.file_utils import DownloadConfig
from .utils.logging import get_logger


logger = get_logger(__name__)


class FileFreeLock(BaseFileLock):
    """Thread lock until a file **cannot** be locked"""

    def __init__(self, lock_file, *args, **kwargs):
        self.filelock = FileLock(lock_file)
        super().__init__(lock_file, *args, **kwargs)

    def _acquire(self):
        try:
            self.filelock.acquire(timeout=0.01, poll_intervall=0.02)  # Try to lock once
        except Timeout:
            # We couldn't acquire the lock, the file is locked!
            self._lock_file_fd = self.filelock.lock_file
        else:
            # We were able to acquire the lock, the file is not yet locked!
            self.filelock.release()
            self._lock_file_fd = None

    def _release(self):
        self._lock_file_fd = None


# lists - summarize long lists similarly to NumPy
# arrays/tensors - let the frameworks control formatting
def summarize_if_long_list(obj):
    if not type(obj) == list or len(obj) <= 6:
        return f"{obj}"

    def format_chunk(chunk):
        return ", ".join(repr(x) for x in chunk)

    return f"[{format_chunk(obj[:3])}, ..., {format_chunk(obj[-3:])}]"


class EvaluationModuleInfoMixin:
    """This base class exposes some attributes of EvaluationModuleInfo
    at the base level of the EvaluationModule for easy access.
    """

    def __init__(self, info: EvaluationModuleInfo):
        self._module_info = info

    @property
    def info(self):
        """:class:`evaluate.EvaluationModuleInfo` object containing all the metadata in the evaluation module."""
        return self._module_info

    @property
    def name(self) -> str:
        return self._module_info.module_name

    @property
    def experiment_id(self) -> Optional[str]:
        return self._module_info.experiment_id

    @property
    def description(self) -> str:
        return self._module_info.description

    @property
    def citation(self) -> str:
        return self._module_info.citation

    @property
    def features(self) -> Features:
        return self._module_info.features

    @property
    def inputs_description(self) -> str:
        return self._module_info.inputs_description

    @property
    def homepage(self) -> Optional[str]:
        return self._module_info.homepage

    @property
    def license(self) -> str:
        return self._module_info.license

    @property
    def codebase_urls(self) -> Optional[List[str]]:
        return self._module_info.codebase_urls

    @property
    def reference_urls(self) -> Optional[List[str]]:
        return self._module_info.reference_urls

    @property
    def streamable(self) -> bool:
        return self._module_info.streamable

    @property
    def format(self) -> Optional[str]:
        return self._module_info.format

    @property
    def module_type(self) -> str:
        return self._module_info.module_type


class EvaluationModule(EvaluationModuleInfoMixin):
    """A EvaluationModule is the base class and common API for metrics, comparisons, and measurements.

    Args:
        config_name (``str``): This is used to define a hash specific to a module computation script and prevents the module's data
            to be overridden when the module loading script is modified.
        keep_in_memory (:obj:`bool`): keep all predictions and references in memory. Not possible in distributed settings.
        cache_dir (``str``): Path to a directory in which temporary prediction/references data will be stored.
            The data directory should be located on a shared file-system in distributed setups.
        num_process (``int``): specify the total number of nodes in a distributed settings.
            This is useful to compute module in distributed setups (in particular non-additive modules like F1).
        process_id (``int``): specify the id of the current process in a distributed setup (between 0 and num_process-1)
            This is useful to compute module in distributed setups (in particular non-additive metrics like F1).
        seed (:obj:`int`, optional): If specified, this will temporarily set numpy's random seed when :func:`evaluate.EvaluationModule.compute` is run.
        experiment_id (``str``): A specific experiment id. This is used if several distributed evaluations share the same file system.
            This is useful to compute module in distributed setups (in particular non-additive metrics like F1).
        max_concurrent_cache_files (``int``): Max number of concurrent module cache files (default 10000).
        timeout (``Union[int, float]``): Timeout in second for distributed setting synchronization.
    """

    def __init__(
        self,
        config_name: Optional[str] = None,
        keep_in_memory: bool = False,
        cache_dir: Optional[str] = None,
        num_process: int = 1,
        process_id: int = 0,
        seed: Optional[int] = None,
        experiment_id: Optional[str] = None,
        max_concurrent_cache_files: int = 10000,
        timeout: Union[int, float] = 100,
        **kwargs,
    ):
        # prepare info
        self.config_name = config_name or "default"
        info = self._info()
        info.module_name = camelcase_to_snakecase(self.__class__.__name__)
        info.config_name = self.config_name
        info.experiment_id = experiment_id or "default_experiment"
        EvaluationModuleInfoMixin.__init__(self, info)  # For easy access on low level

        # Safety checks on num_process and process_id
        if not isinstance(process_id, int) or process_id < 0:
            raise ValueError("'process_id' should be a number greater than 0")
        if not isinstance(num_process, int) or num_process <= process_id:
            raise ValueError("'num_process' should be a number greater than process_id")
        if keep_in_memory and num_process != 1:
            raise ValueError("Using 'keep_in_memory' is not possible in distributed setting (num_process > 1).")

        self.num_process = num_process
        self.process_id = process_id
        self.max_concurrent_cache_files = max_concurrent_cache_files

        self.keep_in_memory = keep_in_memory
        self._data_dir_root = os.path.expanduser(cache_dir or config.HF_METRICS_CACHE)
        self.data_dir = self._build_data_dir()
        if seed is None:
            _, seed, pos, *_ = np.random.get_state()
            self.seed: int = seed[pos] if pos < 624 else seed[0]
        else:
            self.seed: int = seed
        self.timeout: Union[int, float] = timeout
        self.kwargs = kwargs

        # Update 'compute' and 'add' docstring
        # methods need to be copied otherwise it changes the docstrings of every instance
        self.compute = types.MethodType(copyfunc(self.compute), self)
        self.add_batch = types.MethodType(copyfunc(self.add_batch), self)
        self.add = types.MethodType(copyfunc(self.add), self)
        self.compute.__func__.__doc__ += self.info.inputs_description
        self.add_batch.__func__.__doc__ += self.info.inputs_description
        self.add.__func__.__doc__ += self.info.inputs_description

        # self.arrow_schema = pa.schema(field for field in self.info.features.type)
        self.current_features = None
        self.buf_writer = None
        self.writer = None
        self.writer_batch_size = None
        self.data = None

        # This is the cache file we store our predictions/references in
        # Keep it None for now so we can (cloud)pickle the object
        self.cache_file_name = None
        self.filelock = None
        self.rendez_vous_lock = None

        # This is all the cache files on which we have a lock when we are in a distributed setting
        self.file_paths = None
        self.filelocks = None

    def __len__(self):
        """Return the number of examples (predictions or predictions/references pair)
        currently stored in the evaluation module's cache.
        """
        return 0 if self.writer is None else len(self.writer)

    def __repr__(self):
        return (
            f'EvaluationModule(name: "{self.name}", module_type: "{self.module_type}", '
            f'features: {self.features}, usage: """{self.inputs_description}""", '
            f"stored examples: {len(self)})"
        )

    def _build_data_dir(self):
        """Path of this evaluation module in cache_dir:
        Will be:
            self._data_dir_root/self.name/self.config_name/self.hash (if not none)/
        If any of these element is missing or if ``with_version=False`` the corresponding subfolders are dropped.
        """
        builder_data_dir = self._data_dir_root
        builder_data_dir = os.path.join(builder_data_dir, self.name, self.config_name)
        os.makedirs(builder_data_dir, exist_ok=True)
        return builder_data_dir

    def _create_cache_file(self, timeout=1) -> Tuple[str, FileLock]:
        """Create a new cache file. If the default cache file is used, we generated a new hash."""
        file_path = os.path.join(self.data_dir, f"{self.experiment_id}-{self.num_process}-{self.process_id}.arrow")
        filelock = None
        for i in range(self.max_concurrent_cache_files):
            filelock = FileLock(file_path + ".lock")
            try:
                filelock.acquire(timeout=timeout)
            except Timeout:
                # If we have reached the max number of attempts or we are not allow to find a free name (distributed setup)
                # We raise an error
                if self.num_process != 1:
                    raise ValueError(
                        f"Error in _create_cache_file: another evaluation module instance is already using the local cache file at {file_path}. "
                        f"Please specify an experiment_id (currently: {self.experiment_id}) to avoid collision "
                        f"between distributed evaluation module instances."
                    ) from None
                if i == self.max_concurrent_cache_files - 1:
                    raise ValueError(
                        f"Cannot acquire lock, too many evaluation module instance are operating concurrently on this file system."
                        f"You should set a larger value of max_concurrent_cache_files when creating the evaluation module "
                        f"(current value is {self.max_concurrent_cache_files})."
                    ) from None
                # In other cases (allow to find new file name + not yet at max num of attempts) we can try to sample a new hashing name.
                file_uuid = str(uuid.uuid4())
                file_path = os.path.join(
                    self.data_dir, f"{self.experiment_id}-{file_uuid}-{self.num_process}-{self.process_id}.arrow"
                )
            else:
                break

        return file_path, filelock

    def _get_all_cache_files(self) -> Tuple[List[str], List[FileLock]]:
        """Get a lock on all the cache files in a distributed setup.
        We wait for timeout second to let all the distributed node finish their tasks (default is 100 seconds).
        """
        if self.num_process == 1:
            if self.cache_file_name is None:
                raise ValueError(
                    "Evaluation module cache file doesn't exist. Please make sure that you call `add` or `add_batch` "
                    "at least once before calling `compute`."
                )
            file_paths = [self.cache_file_name]
        else:
            file_paths = [
                os.path.join(self.data_dir, f"{self.experiment_id}-{self.num_process}-{process_id}.arrow")
                for process_id in range(self.num_process)
            ]

        # Let's acquire a lock on each process files to be sure they are finished writing
        filelocks = []
        for process_id, file_path in enumerate(file_paths):
            if process_id == 0:  # process 0 already has its lock file
                filelocks.append(self.filelock)
            else:
                filelock = FileLock(file_path + ".lock")
                try:
                    filelock.acquire(timeout=self.timeout)
                except Timeout:
                    raise ValueError(
                        f"Cannot acquire lock on cached file {file_path} for process {process_id}."
                    ) from None
                else:
                    filelocks.append(filelock)

        return file_paths, filelocks

    def _check_all_processes_locks(self):
        expected_lock_file_names = [
            os.path.join(self.data_dir, f"{self.experiment_id}-{self.num_process}-{process_id}.arrow.lock")
            for process_id in range(self.num_process)
        ]
        for expected_lock_file_name in expected_lock_file_names:
            nofilelock = FileFreeLock(expected_lock_file_name)
            try:
                nofilelock.acquire(timeout=self.timeout)
            except Timeout:
                raise ValueError(
                    f"Expected to find locked file {expected_lock_file_name} from process {self.process_id} but it doesn't exist."
                ) from None
            else:
                nofilelock.release()

    def _check_rendez_vous(self):
        expected_lock_file_name = os.path.join(self.data_dir, f"{self.experiment_id}-{self.num_process}-0.arrow.lock")
        nofilelock = FileFreeLock(expected_lock_file_name)
        try:
            nofilelock.acquire(timeout=self.timeout)
        except Timeout:
            raise ValueError(
                f"Expected to find locked file {expected_lock_file_name} from process {self.process_id} but it doesn't exist."
            ) from None
        else:
            nofilelock.release()
        lock_file_name = os.path.join(self.data_dir, f"{self.experiment_id}-{self.num_process}-rdv.lock")
        rendez_vous_lock = FileLock(lock_file_name)
        try:
            rendez_vous_lock.acquire(timeout=self.timeout)
        except Timeout:
            raise ValueError(f"Couldn't acquire lock on {lock_file_name} from process {self.process_id}.") from None
        else:
            rendez_vous_lock.release()

    def _finalize(self):
        """Close all the writing process and load/gather the data
        from all the nodes if main node or all_process is True.
        """
        if self.writer is not None:
            self.writer.finalize()
        self.writer = None
        # release the locks of the processes > 0 so that process 0 can lock them to read + delete the data
        if self.filelock is not None and self.process_id > 0:
            self.filelock.release()

        if self.keep_in_memory:
            # Read the predictions and references
            reader = ArrowReader(path=self.data_dir, info=DatasetInfo(features=self.current_features))
            self.data = Dataset.from_buffer(self.buf_writer.getvalue())

        elif self.process_id == 0:
            # Let's acquire a lock on each node files to be sure they are finished writing
            file_paths, filelocks = self._get_all_cache_files()

            # Read the predictions and references
            try:
                reader = ArrowReader(path="", info=DatasetInfo(features=self.current_features))
                self.data = Dataset(**reader.read_files([{"filename": f} for f in file_paths]))
            except FileNotFoundError:
                raise ValueError(
                    "Error in finalize: another evaluation module instance is already using the local cache file. "
                    "Please specify an experiment_id to avoid collision between distributed evaluation module instances."
                ) from None

            # Store file paths and locks and we will release/delete them after the computation.
            self.file_paths = file_paths
            self.filelocks = filelocks

    def compute(self, *, predictions=None, references=None, **kwargs) -> Optional[dict]:
        """Compute the evaluation module.

        Usage of positional arguments is not allowed to prevent mistakes.

        Args:
            predictions (list/array/tensor, optional): Predictions.
            references (list/array/tensor, optional): References.
            **kwargs (optional): Keyword arguments that will be forwarded to the evaluation module :meth:`_compute`
                method (see details in the docstring).

        Return:
            dict or None

            - Dictionary with the results if this evaluation module is run on the main process (``process_id == 0``).
            - None if the evaluation module is not run on the main process (``process_id != 0``).
        """
        all_kwargs = {"predictions": predictions, "references": references, **kwargs, **self.kwargs}
        if predictions is None and references is None:
            missing_kwargs = {k: None for k in self._feature_names() if k not in all_kwargs}
            all_kwargs.update(missing_kwargs)
        else:
            missing_inputs = [k for k in self._feature_names() if k not in all_kwargs]
            if missing_inputs:
                raise ValueError(
                    f"Evaluation module inputs are missing: {missing_inputs}. All required inputs are {list(self._feature_names())}"
                )
        inputs = {input_name: all_kwargs[input_name] for input_name in self._feature_names()}
        compute_kwargs = {k: kwargs[k] for k in kwargs if k not in self._feature_names()}
        self_kwargs = {k: self.kwargs[k] for k in self.kwargs if k not in self._feature_names()}
        compute_kwargs.update(self_kwargs)

        if any(v is not None for v in inputs.values()):
            self.add_batch(**inputs)
        self._finalize()

        self.cache_file_name = None
        self.filelock = None
        self.current_features = None

        if self.process_id == 0:
            self.data.set_format(type=self.info.format)

            inputs = {input_name: self.data[input_name] for input_name in self._feature_names()}
            with temp_seed(self.seed):
                output = self._compute(**inputs, **compute_kwargs)

            if self.buf_writer is not None:
                self.buf_writer = None
                del self.data
                self.data = None
            else:
                # Release locks and delete all the cache files. Process 0 is released last.
                for filelock, file_path in reversed(list(zip(self.filelocks, self.file_paths))):
                    logger.info(f"Removing {file_path}")
                    del self.data
                    self.data = None
                    del self.writer
                    self.writer = None
                    os.remove(file_path)
                    filelock.release()

            return output
        else:
            return None

    def add_batch(self, *, predictions=None, references=None, **kwargs):
        """Add a batch of predictions and references for the evaluation module's stack.

        Args:
            predictions (list/array/tensor, optional): Predictions.
            references (list/array/tensor, optional): References.
        """
        bad_inputs = [input_name for input_name in kwargs if input_name not in self._feature_names()]
        if bad_inputs:
            raise ValueError(
                f"Bad inputs for evaluation module: {bad_inputs}. All required inputs are {list(self._feature_names())}"
            )
        batch = {"predictions": predictions, "references": references, **kwargs}
        batch = {intput_name: batch[intput_name] for intput_name in self._feature_names()}
        if self.writer is None:
            self.current_features = self._infer_feature_from_batch(batch)
            self._init_writer()
        try:
            for key, column in batch.items():
                [self._enforce_nested_string_type(self.current_features[key], obj) for obj in column]
            batch = self.current_features.encode_batch(batch)
            self.writer.write_batch(batch)
        except (pa.ArrowInvalid, TypeError):
            if any(len(batch[c]) != len(next(iter(batch.values()))) for c in batch):
                col0 = next(iter(batch))
                bad_col = [c for c in batch if len(batch[c]) != len(batch[col0])][0]
                error_msg = (
                    f"Mismatch in the number of {col0} ({len(batch[col0])}) and {bad_col} ({len(batch[bad_col])})"
                )
            elif sorted(self.current_features) != ["references", "predictions"]:
                error_msg = (
                    f"Module inputs don't match the expected format.\n" f"Expected format: {self.current_features },\n"
                )
                error_msg_inputs = ",\n".join(
                    f"Input {input_name}: {summarize_if_long_list(batch[input_name])}"
                    for input_name in self.current_features
                )
                error_msg += error_msg_inputs
            else:
                error_msg = (
                    f"Predictions and/or references don't match the expected format.\n"
                    f"Expected format: {self.current_features },\n"
                    f"Input predictions: {summarize_if_long_list(predictions)},\n"
                    f"Input references: {summarize_if_long_list(references)}"
                )
            raise ValueError(error_msg) from None

    def add(self, *, prediction=None, reference=None, **kwargs):
        """Add one prediction and reference for the evaluation module's stack.

        Args:
            prediction (list/array/tensor, optional): Predictions.
            reference (list/array/tensor, optional): References.
        """
        bad_inputs = [input_name for input_name in kwargs if input_name not in self._feature_names()]
        if bad_inputs:
            raise ValueError(
                f"Bad inputs for evaluation module: {bad_inputs}. All required inputs are {list(self._feature_names())}"
            )
        example = {"predictions": prediction, "references": reference, **kwargs}
        example = {intput_name: example[intput_name] for intput_name in self._feature_names()}
        if self.writer is None:
            self.current_features = self._infer_feature_from_example(example)
            self._init_writer()
        try:
            self._enforce_nested_string_type(self.info.features, example)
            example = self.info.features.encode_example(example)
            self.writer.write(example)
        except (pa.ArrowInvalid, TypeError):
            error_msg = (
                f"Evaluation module inputs don't match the expected format.\n"
                f"Expected format: {self.current_features},\n"
            )
            error_msg_inputs = ",\n".join(
                f"Input {input_name}: {summarize_if_long_list(example[input_name])}"
                for input_name in self.current_features
            )
            error_msg += error_msg_inputs
            raise ValueError(error_msg) from None

    def _infer_feature_from_batch(self, batch):
        if isinstance(self.features, Features):
            return self.features
        else:
            example = dict([(k, v[0]) for k, v in batch.items()])
            return self._infer_feature_from_example(example)

    def _infer_feature_from_example(self, example):
        if isinstance(self.features, Features):
            return self.features
        else:
            for features in self.features:
                try:
                    self._enforce_nested_string_type(features, example)
                    features.encode_example(example)
                    return features
                except ValueError:
                    continue
        feature_strings = "\n".join([f"Feature option {i}: {feature}" for i, feature in enumerate(self.features)])
        error_msg = (
            f"Predictions and/or references don't match the expected format.\n"
            f"Expected format:\n{feature_strings},\n"
            f"Input predictions: {summarize_if_long_list(example['predictions'])},\n"
            f"Input references: {summarize_if_long_list(example['references'])}"
        )
        raise ValueError(error_msg) from None

    def _feature_names(self):
        if isinstance(self.features, list):
            feature_names = list(self.features[0].keys())
        else:
            feature_names = list(self.features.keys())
        return feature_names

    def _init_writer(self, timeout=1):
        if self.num_process > 1:
            if self.process_id == 0:
                file_path = os.path.join(self.data_dir, f"{self.experiment_id}-{self.num_process}-rdv.lock")
                self.rendez_vous_lock = FileLock(file_path)
                try:
                    self.rendez_vous_lock.acquire(timeout=timeout)
                except TimeoutError:
                    raise ValueError(
                        f"Error in _init_writer: another evalution module instance is already using the local cache file at {file_path}. "
                        f"Please specify an experiment_id (currently: {self.experiment_id}) to avoid collision "
                        f"between distributed evaluation module instances."
                    ) from None

        if self.keep_in_memory:
            self.buf_writer = pa.BufferOutputStream()
            self.writer = ArrowWriter(
                features=self.current_features, stream=self.buf_writer, writer_batch_size=self.writer_batch_size
            )
        else:
            self.buf_writer = None

            # Get cache file name and lock it
            if self.cache_file_name is None or self.filelock is None:
                cache_file_name, filelock = self._create_cache_file()  # get ready
                self.cache_file_name = cache_file_name
                self.filelock = filelock

            self.writer = ArrowWriter(
                features=self.current_features, path=self.cache_file_name, writer_batch_size=self.writer_batch_size
            )
        # Setup rendez-vous here if
        if self.num_process > 1:
            if self.process_id == 0:
                self._check_all_processes_locks()  # wait for everyone to be ready
                self.rendez_vous_lock.release()  # let everyone go
            else:
                self._check_rendez_vous()  # wait for master to be ready and to let everyone go

    def _info(self) -> EvaluationModuleInfo:
        """Construct the EvaluationModuleInfo object. See `EvaluationModuleInfo` for details.

        Warning: This function is only called once and the result is cached for all
        following .info() calls.

        Returns:
            info: (EvaluationModuleInfo) The EvaluationModule information
        """
        raise NotImplementedError

    def download_and_prepare(
        self,
        download_config: Optional[DownloadConfig] = None,
        dl_manager: Optional[DownloadManager] = None,
    ):
        """Downloads and prepares dataset for reading.

        Args:
            download_config (:class:`DownloadConfig`, optional): Specific download configuration parameters.
            dl_manager (:class:`DownloadManager`, optional): Specific download manager to use.
        """
        if dl_manager is None:
            if download_config is None:
                download_config = DownloadConfig()
                download_config.cache_dir = os.path.join(self.data_dir, "downloads")
                download_config.force_download = False

            dl_manager = DownloadManager(
                dataset_name=self.name, download_config=download_config, data_dir=self.data_dir
            )

        self._download_and_prepare(dl_manager)

    def _download_and_prepare(self, dl_manager):
        """Downloads and prepares resources for the evaluation module.

        This is the internal implementation to overwrite called when user calls
        `download_and_prepare`. It should download all required resources for the evaluation module.

        Args:
            dl_manager (:class:`DownloadManager`): `DownloadManager` used to download and cache data.
        """
        return None

    def _compute(self, *, predictions=None, references=None, **kwargs) -> Dict[str, Any]:
        """This method defines the common API for all the evaluation module in the library"""
        raise NotImplementedError

    def __del__(self):
        if hasattr(self, "filelock") and self.filelock is not None:
            self.filelock.release()
        if hasattr(self, "rendez_vous_lock") and self.rendez_vous_lock is not None:
            self.rendez_vous_lock.release()
        if hasattr(self, "writer"):  # in case it was already deleted
            del self.writer
        if hasattr(self, "data"):  # in case it was already deleted
            del self.data

    def _enforce_nested_string_type(self, schema, obj):
        """
        Recursively checks if there is any Value feature of type string and throws TypeError if corresponding object is not a string.
        Since any Python object can be cast to string this avoids implicitly casting wrong input types (e.g. lists) to string without error.
        """
        # Nested structures: we allow dict, list, tuples, sequences
        if isinstance(schema, dict):
            return [self._enforce_nested_string_type(sub_schema, o) for k, (sub_schema, o) in zip_dict(schema, obj)]

        elif isinstance(schema, (list, tuple)):
            sub_schema = schema[0]
            return [self._enforce_nested_string_type(sub_schema, o) for o in obj]
        elif isinstance(schema, Sequence):
            # We allow to reverse list of dict => dict of list for compatiblity with tfds
            if isinstance(schema.feature, dict):
                if isinstance(obj, (list, tuple)):
                    # obj is a list of dict
                    for k, dict_tuples in zip_dict(schema.feature, *obj):
                        return [self._enforce_nested_string_type(dict_tuples[0], o) for o in dict_tuples[1:]]
                else:
                    # obj is a single dict
                    for k, (sub_schema, sub_objs) in zip_dict(schema.feature, obj):
                        return [self._enforce_nested_string_type(sub_schema, o) for o in sub_objs]
            # schema.feature is not a dict
            if isinstance(obj, str):  # don't interpret a string as a list
                raise ValueError(f"Got a string but expected a list instead: '{obj}'")
            if obj is None:
                return None
            else:
                if len(obj) > 0:
                    for first_elmt in obj:
                        if _check_non_null_non_empty_recursive(first_elmt, schema.feature):
                            break
                    if not isinstance(first_elmt, list):
                        return [self._enforce_nested_string_type(schema.feature, o) for o in obj]

        elif isinstance(schema, Value):
            if pa.types.is_string(schema.pa_type) and not isinstance(obj, str):
                raise TypeError(f"Expected type str but got {type(obj)}.")

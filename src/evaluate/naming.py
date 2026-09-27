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
"""Utilities for file names."""

import itertools
import os
import re


_uppercase_uppercase_re = re.compile(r"([A-Z]+)([A-Z][a-z])")
_lowercase_uppercase_re = re.compile(r"([a-z\d])([A-Z])")

_single_underscore_re = re.compile(r"(?<!_)_(?!_)")
_multiple_underscores_re = re.compile(r"(_{2,})")

_split_re = r"^\w+(\.\w+)*$"


def camelcase_to_snakecase(name):
    """Convert camel-case string to snake-case."""
    name = _uppercase_uppercase_re.sub(r"\1_\2", name)
    name = _lowercase_uppercase_re.sub(r"\1_\2", name)
    return name.lower()


def snakecase_to_camelcase(name):
    """Convert snake-case string to camel-case string."""
    name = _single_underscore_re.split(name)
    name = [_multiple_underscores_re.split(n) for n in name]
    return "".join(n.capitalize() for n in itertools.chain.from_iterable(name) if n != "")


def filename_prefix_for_name(name):
    """Return the snake_case filename prefix for a given dataset name.

    Args:
        name (`str`): Dataset name (must not contain path separators).

    Returns:
        `str`: The snake_case version of `name`, used as the base prefix for data files.

    Raises:
        `ValueError`: If `name` looks like a file path rather than a plain dataset name.

    Example:
        ```py
        >>> filename_prefix_for_name("MyDataset")
        'my_dataset'
        ```
    """
    if os.path.basename(name) != name:
        raise ValueError(f"Should be a dataset name, not a path: {name}")
    return camelcase_to_snakecase(name)


def filename_prefix_for_split(name, split):
    """Return the filename prefix for a specific dataset split.

    Combines the snake_case dataset name with the split name, following
    the pattern ``<snake_case_name>-<split>``.

    Args:
        name (`str`): Dataset name (must not contain path separators).
        split (`str`): Split identifier, e.g. ``"train"``, ``"test"``, or
            a dotted sub-split such as ``"train.clean"``.

    Returns:
        `str`: The filename prefix in the form ``<name_prefix>-<split>``.

    Raises:
        `ValueError`: If `name` looks like a path, or `split` does not match
            the expected pattern ``^\\w+(\\.\\w+)*$``.

    Example:
        ```py
        >>> filename_prefix_for_split("MyDataset", "train")
        'my_dataset-train'
        ```
    """
    if os.path.basename(name) != name:
        raise ValueError(f"Should be a dataset name, not a path: {name}")
    if not re.match(_split_re, split):
        raise ValueError(f"Split name should match '{_split_re}'' but got '{split}'.")
    return f"{filename_prefix_for_name(name)}-{split}"


def filepattern_for_dataset_split(dataset_name, split, data_dir, filetype_suffix=None):
    """Return a glob pattern matching all shard files for a dataset split.

    Args:
        dataset_name (`str`): Dataset name (must not contain path separators).
        split (`str`): Split identifier, e.g. ``"train"`` or ``"validation"``.
        data_dir (`str`): Directory where the dataset files are stored.
        filetype_suffix (`str`, *optional*): File extension to append before the
            glob wildcard, e.g. ``"parquet"`` → ``"my_dataset-train.parquet*"``.
            When ``None`` the pattern has no extension suffix.

    Returns:
        `str`: A glob pattern of the form ``<data_dir>/<prefix>[.<suffix>]*``.

    Example:
        ```py
        >>> filepattern_for_dataset_split("MyDataset", "train", "/data", "parquet")
        '/data/my_dataset-train.parquet*'
        ```
    """
    prefix = filename_prefix_for_split(dataset_name, split)
    if filetype_suffix:
        prefix += f".{filetype_suffix}"
    filepath = os.path.join(data_dir, prefix)
    return f"{filepath}*"


def filename_for_dataset_split(dataset_name, split, filetype_suffix=None):
    """Return the filename (without directory) for a dataset split.

    Args:
        dataset_name (`str`): Dataset name (must not contain path separators).
        split (`str`): Split identifier, e.g. ``"train"`` or ``"test"``.
        filetype_suffix (`str`, *optional*): File extension to append, e.g.
            ``"arrow"`` → ``"my_dataset-train.arrow"``. When ``None`` no
            extension is added.

    Returns:
        `str`: The filename string, e.g. ``"my_dataset-train.arrow"``.

    Example:
        ```py
        >>> filename_for_dataset_split("MyDataset", "train", "arrow")
        'my_dataset-train.arrow'
        ```
    """
    prefix = filename_prefix_for_split(dataset_name, split)
    if filetype_suffix:
        prefix += f".{filetype_suffix}"
    return prefix


def filepath_for_dataset_split(dataset_name, split, data_dir, filetype_suffix=None):
    """Return the full file path for a dataset split file.

    Combines `data_dir` and the result of :func:`filename_for_dataset_split`
    to produce an absolute-or-relative path suitable for reading or writing.

    Args:
        dataset_name (`str`): Dataset name (must not contain path separators).
        split (`str`): Split identifier, e.g. ``"train"`` or ``"validation"``.
        data_dir (`str`): Directory where the dataset files are stored.
        filetype_suffix (`str`, *optional*): File extension without the leading
            dot, e.g. ``"arrow"``. When ``None`` no extension is added.

    Returns:
        `str`: Full path of the form ``<data_dir>/<filename>``.

    Example:
        ```py
        >>> filepath_for_dataset_split("MyDataset", "train", "/data", "arrow")
        '/data/my_dataset-train.arrow'
        ```
    """
    filename = filename_for_dataset_split(
        dataset_name=dataset_name,
        split=split,
        filetype_suffix=filetype_suffix,
    )
    filepath = os.path.join(data_dir, filename)
    return filepath

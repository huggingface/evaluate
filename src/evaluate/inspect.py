# Copyright 2020 The HuggingFace Evaluate Authors.
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
""" List and inspect metrics."""

from typing import Optional

import requests
from datasets.utils import DownloadConfig

from .config import EVALUATION_MODULE_TYPES, HF_LIST_ENDPOINT
from .loading import evaluation_module_factory
from .utils.logging import get_logger


logger = get_logger(__name__)


class SplitsNotFoundError(ValueError):
    pass


def list_evaluation_modules(module_type=None, include_community=True, with_details=False):
    """List all evaluation modules available on the Hugging Face Hub.

    Args:
        module_type (`str`, *optional*, default `None`): Type of evaluation modules to list. Has to be one of `'metric'`, `'comparison'`, or `'measurement'`. If `None` all types are listed.
        include_community (`bool`, *optional*, default `True`): Include community modules in the list.
        with_details (`bool`, *optional*, default `False`): Return the full details on the metrics instead of only the ID.

    Returns:
        `List[Union[str, dict]]`
    """

    if module_type is None:
        evaluations_list = []
        for module_type in EVALUATION_MODULE_TYPES:
            evaluations_list.extend(
                _list_evaluation_modules_type(
                    module_type, include_community=include_community, with_details=with_details
                )
            )
    else:
        if module_type not in EVALUATION_MODULE_TYPES:
            raise ValueError(f"Invalid module type '{module_type}'. Has to be one of {EVALUATION_MODULE_TYPES}.")
        evaluations_list = _list_evaluation_modules_type(
            module_type, include_community=include_community, with_details=with_details
        )
    return evaluations_list


def _list_evaluation_modules_type(module_type, include_community=True, with_details=False):

    r = requests.get(HF_LIST_ENDPOINT.format(type=module_type))
    r.raise_for_status()
    d = r.json()

    if not include_community:
        d = [element for element in d if element["id"].split("/")[0] == f"evaluate-{module_type}"]

    # remove namespace for canonical modules and add community tag
    for element in d:
        if element["id"].split("/")[0] == f"evaluate-{module_type}":
            element["id"] = element["id"].split("/")[1]
            element["community"] = False
        else:
            element["community"] = True

    if with_details:
        return [
            {
                "name": element["id"],
                "type": module_type,
                "community": element["community"],
                "likes": element.get("likes", 0),
            }
            for element in d
        ]
    else:
        return [element["id"] for element in d]


def inspect_evaluation_module(
    path: str, local_path: str, download_config: Optional[DownloadConfig] = None, **download_kwargs
):
    r"""
    Allow inspection/modification of a evaluation script by copying it on local drive at local_path.

    Args:
        path (``str``): path to the evaluation script. Can be either:

            - a local path to script or the directory containing the script (if the script has the same name as the directory),
                e.g. ``'./metrics/accuracy'`` or ``'./metrics/accuracy/accuracy.py'``
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with ``evaluate.list_evaluation_modules()``)
                e.g. ``'accuracy'``, ``'bleu'`` or ``'word_length'``
        local_path (``str``): path to the local folder to copy the datset script to.
        download_config (Optional ``datasets.DownloadConfig``: specific download configuration parameters.
        **download_kwargs: optional attributes for DownloadConfig() which will override the attributes in download_config if supplied.
    """
    evaluation_module = evaluation_module_factory(
        path, download_config=download_config, force_local_path=local_path, **download_kwargs
    )
    print(
        f"The processing scripts for metric {path} can be inspected at {local_path}. "
        f"The main class is in {evaluation_module.module_path}. "
        f"You can modify this processing scripts and use it with `evaluate.load({local_path})`."
    )

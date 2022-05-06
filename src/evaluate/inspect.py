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

import huggingface_hub
from datasets.utils import DownloadConfig

from .load import evaluate_module_factory
from .utils.logging import get_logger


logger = get_logger(__name__)


class SplitsNotFoundError(ValueError):
    pass


def list_metrics(with_community_metrics=True, with_details=False):
    """List all the metrics script available on the Hugging Face Hub.

    Args:
        with_community_metrics (:obj:`bool`, optional, default ``True``): Include the community provided metrics.
        with_details (:obj:`bool`, optional, default ``False``): Return the full details on the metrics instead of only the short name.
    """
    metrics = huggingface_hub.list_metrics()
    if not with_community_metrics:
        metrics = [metric for metric in metrics if "/" not in metric.id]
    if not with_details:
        metrics = [metric.id for metric in metrics]
    return metrics


def inspect_metric(path: str, local_path: str, download_config: Optional[DownloadConfig] = None, **download_kwargs):
    r"""
    Allow inspection/modification of a metric script by copying it on local drive at local_path.

    Args:
        path (``str``): path to the dataset processing script with the dataset builder. Can be either:

            - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
                e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'``
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with ``datasets.list_datasets()``)
                e.g. ``'squad'``, ``'glue'`` or ``'openai/webtext'``
        local_path (``str``): path to the local folder to copy the datset script to.
        download_config (Optional ``datasets.DownloadConfig``: specific download configuration parameters.
        **download_kwargs: optional attributes for DownloadConfig() which will override the attributes in download_config if supplied.
    """
    metric_module = evaluate_module_factory(
        path,
        module_namespace="metrics",
        download_config=download_config,
        force_local_path=local_path,
        **download_kwargs,
    )
    print(
        f"The processing scripts for metric {path} can be inspected at {local_path}. "
        f"The main class is in {metric_module.module_path}. "
        f"You can modify this processing scripts and use it with `evaluate.load_metric({local_path})`."
    )

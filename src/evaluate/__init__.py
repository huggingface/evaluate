# flake8: noqa
# Copyright 2020 The HuggingFace Evaluate Authors and the TensorFlow Datasets Authors.
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
# pylint: enable=line-too-long
# pylint: disable=g-import-not-at-top,g-bad-import-order,wrong-import-position

__version__ = "0.0.1.dev0"

from packaging import version


SCRIPTS_VERSION = "main" if version.parse(__version__).is_devrelease else __version__

del version

from typing import Dict, List

from datasets import load_metric
from datasets.metric import Metric
from transformers.pipelines import SUPPORTED_TASKS as SUPPORTED_PIPELINE_TASKS
from transformers.pipelines import TASK_ALIASES
from transformers.pipelines import check_task as check_pipeline_task

from .evaluator import Evaluator, TextClassificationEvaluator
from .info import EvaluationModuleInfo
from .inspect import inspect_metric, list_metrics
from .loading import load
from .module import EvaluationModule
from .saving import save
from .utils import *
from .utils import gradio, logging


SUPPORTED_EVALUATOR_TASKS = {
    "text-classification": {
        "impl": TextClassificationEvaluator,
        "default_metric": "f1",
    }
}


def get_supported_tasks() -> List[str]:
    return SUPPORTED_EVALUATOR_TASKS.keys()


def check_task(task: str) -> Dict:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.
    """
    if task in TASK_ALIASES:
        task = TASK_ALIASES[task]
    if not check_pipeline_task(task):
        raise KeyError(f"Unknown task {task}, available tasks are {get_supported_tasks()}")
    pipeline_tasks = SUPPORTED_PIPELINE_TASKS.keys()
    if task in SUPPORTED_EVALUATOR_TASKS.keys() and task in pipeline_tasks:
        return SUPPORTED_EVALUATOR_TASKS[task]
    raise KeyError(f"Unknown task {task}, available tasks are {get_supported_tasks()}")


def evaluator(task: str = None, default_metric: Metric = None) -> Evaluator:
    targeted_task = check_task(task)
    evaluator_class = targeted_task["impl"]
    default_metric = load_metric(targeted_task["default_metric"])
    return evaluator_class(task=task, default_metric=default_metric)

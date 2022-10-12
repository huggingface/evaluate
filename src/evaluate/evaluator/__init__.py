# Copyright 2022 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
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


try:
    from transformers.pipelines import SUPPORTED_TASKS as SUPPORTED_PIPELINE_TASKS
    from transformers.pipelines import TASK_ALIASES
    from transformers.pipelines import check_task as check_pipeline_task

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from typing import Dict, List

from .base import Evaluator
from .image_classification import ImageClassificationEvaluator
from .question_answering import QuestionAnsweringEvaluator
from .text2text_generation import SummarizationEvaluator, Text2TextGenerationEvaluator, TranslationEvaluator
from .text_classification import TextClassificationEvaluator
from .token_classification import TokenClassificationEvaluator


SUPPORTED_EVALUATOR_TASKS = {
    "text-classification": {
        "implementation": TextClassificationEvaluator,
        "default_metric_name": "accuracy",
    },
    "image-classification": {
        "implementation": ImageClassificationEvaluator,
        "default_metric_name": "accuracy",
    },
    "question-answering": {
        "implementation": QuestionAnsweringEvaluator,
        "default_metric_name": "squad",
    },
    "token-classification": {
        "implementation": TokenClassificationEvaluator,
        "default_metric_name": "seqeval",
    },
    "text2text-generation": {
        "implementation": Text2TextGenerationEvaluator,
        "default_metric_name": "bleu",
    },
    "summarization": {
        "implementation": SummarizationEvaluator,
        "default_metric_name": "rouge",
    },
    "translation": {
        "implementation": TranslationEvaluator,
        "default_metric_name": "bleu",
    },
}


def get_supported_tasks() -> List[str]:
    """
    Returns a list of supported task strings.
    """
    return list(SUPPORTED_EVALUATOR_TASKS.keys())


def check_task(task: str) -> Dict:
    """
    Checks an incoming task string, to validate it's correct and returns the default Evaluator class and default metric
    name. It first performs a check to validata that the string is a valid `Pipeline` task, then it checks if it's a
    valid `Evaluator` task. `Evaluator` tasks are a substet of `Pipeline` tasks.
    Args:
        task (`str`):
            The task defining which evaluator will be returned. Currently accepted tasks are:
            - `"image-classification"`
            - `"question-answering"`
            - `"text-classification"` (alias `"sentiment-analysis"` available)
            - `"token-classification"`
    Returns:
        task_defaults: `dict`, contains the implementasion class of a give Evaluator and the default metric name.
    """
    if task in TASK_ALIASES:
        task = TASK_ALIASES[task]
    if not check_pipeline_task(task):
        raise KeyError(f"Unknown task {task}, available tasks are: {get_supported_tasks()}.")
    if task in SUPPORTED_EVALUATOR_TASKS.keys() and task in SUPPORTED_PIPELINE_TASKS.keys():
        return SUPPORTED_EVALUATOR_TASKS[task]
    raise KeyError(f"Unknown task {task}, available tasks are: {get_supported_tasks()}.")


def evaluator(task: str = None) -> Evaluator:
    """
    Utility factory method to build an [`Evaluator`].
    Evaluators encapsulate a task and a default metric name. They leverage `pipeline` functionalify from `transformers`
    to simplify the evaluation of multiple combinations of models, datasets and metrics for a given task.
    Args:
        task (`str`):
            The task defining which evaluator will be returned. Currently accepted tasks are:
            - `"image-classification"`: will return a [`ImageClassificationEvaluator`].
            - `"question-answering"`: will return a [`QuestionAnsweringEvaluator`].
            - `"text-classification"` (alias `"sentiment-analysis"` available): will return a [`TextClassificationEvaluator`].
            - `"token-classification"`: will return a [`TokenClassificationEvaluator`].
    Returns:
        [`Evaluator`]: An evaluator suitable for the task.
    Examples:
    ```python
    >>> from evaluate import evaluator
    >>> # Sentiment analysis evaluator
    >>> evaluator("sentiment-analysis")
    ```"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "If you want to use the `Evaluator` you need `transformers`. Run `pip install evaluate[transformers]`."
        )
    targeted_task = check_task(task)
    evaluator_class = targeted_task["implementation"]
    default_metric_name = targeted_task["default_metric_name"]
    return evaluator_class(task=task, default_metric_name=default_metric_name)

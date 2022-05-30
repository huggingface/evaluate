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

from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Lint as: python3
from datasets import Dataset, load_dataset
from scipy.stats import bootstrap
from transformers import Pipeline, PreTrainedModel, PreTrainedTokenizer, TFPreTrainedModel, pipeline
from transformers.pipelines import SUPPORTED_TASKS as SUPPORTED_PIPELINE_TASKS
from transformers.pipelines import TASK_ALIASES
from transformers.pipelines import check_task as check_pipeline_task
from typing_extensions import Literal

from .loading import load
from .module import EvaluationModule
from .utils.logging import get_logger


logger = get_logger(__name__)


class Evaluator(ABC):
    def __init__(self, task: str, default_metric_name: str = None):
        self.task = task
        self.default_metric_name = default_metric_name

    @abstractmethod
    def _compute_predictions(self, pipe: Pipeline, inputs, **predictions_parameters: Dict):
        raise NotImplementedError()

    @staticmethod
    def _compute_confidence_interval(
        predictions,
        references,
        metric: EvaluationModule,
        metric_keys: List[str],
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        bootstrap_dict = {}
        for key in metric_keys:
            bootstrap_dict[key] = bootstrap(
                data=(predictions, references),
                statistic=lambda predictions, references: metric.compute(
                    predictions=predictions,
                    references=references,
                )[key],
                paired=True,
                vectorized=False,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                random_state=random_state,
            )
        return bootstrap_dict

    @abstractmethod
    def compute(
        self,
        model_or_pipeline: Union[str, Pipeline, Callable, PreTrainedModel, TFPreTrainedModel] = None,
        data: Union[str, Dataset] = None,
        metric: Union[str, EvaluationModule] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        **compute_parameters: Dict,
    ):
        raise NotImplementedError()


class TextClassificationEvaluator(Evaluator):
    def __init__(self, task="text-classification", default_metric_name=None):
        super().__init__(task, default_metric_name=default_metric_name)

    def _compute_predictions(self, pipe: Pipeline, inputs, label_mapping: Dict[str, Number] = None) -> List[Number]:
        predictions = pipe(inputs, truncation=True)
        return [
            label_mapping[element["label"]] if label_mapping is not None else element["label"]
            for element in predictions
        ]

    def compute(
        self,
        model_or_pipeline: Union[str, Pipeline, Callable, PreTrainedModel, TFPreTrainedModel] = None,
        data: Union[str, Dataset] = None,
        metric: Union[str, EvaluationModule] = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        random_state: Optional[int] = None,
        input_column: str = "inputs",
        label_column: str = "references",
        label_mapping: Optional[Dict[str, Number]] = None,
    ) -> Tuple[Dict[str, float], Any]:
        # Prepare data.
        if data is None:
            raise ValueError(
                "Please specify a valid `data` object - either a `str` with a name or a `Dataset` object."
            )
        data = load_dataset(data) if isinstance(data, str) else data
        if input_column not in data.column_names:
            raise ValueError(
                f"Invalid `input_column` {input_column} specified. The dataset contains the following columns: {data.column_names}."
            )
        if label_column not in data.column_names:
            raise ValueError(
                f"Invalid `label_column` {label_column} specified. The dataset contains the following columns: {data.column_names}."
            )

        # Prepare pipeline.
        if (
            isinstance(model_or_pipeline, PreTrainedModel)
            or isinstance(model_or_pipeline, TFPreTrainedModel)
            or isinstance(model_or_pipeline, str)
        ):
            pipe = pipeline(self.task, model=model_or_pipeline, tokenizer=tokenizer)
        else:
            if model_or_pipeline is None:
                pipe = pipeline(self.task)
            else:
                pipe = model_or_pipeline
            if tokenizer is not None:
                logger.warning("Ignoring the value of the `tokenizer` argument.")
        if pipe.task != self.task:
            raise ValueError(
                f"Incompatible `model_or_pipeline`. Please specify `model_or_pipeline` compatible with the `{self.task}` task."
            )

        # Prepare metric.
        if metric is None:
            if self.default_metric_name is None:
                raise ValueError(
                    f"`Evaluator` doesn't specify a default metric. Please specify a valid `metric` argument."
                )
            metric = load(self.default_metric_name)
        elif isinstance(metric, str):
            metric = load(metric)

        # Core computations.
        references = data[label_column]
        predictions = self._compute_predictions(pipe, data[input_column], label_mapping=label_mapping)
        result = metric.compute(predictions=predictions, references=references)

        bootstrap = (
            Evaluator._compute_confidence_interval(
                predictions,
                references,
                metric,
                result.keys(),
                confidence_level,
                n_resamples,
                random_state,
            )
            if strategy == "bootstrap"
            else None
        )
        return result, bootstrap


SUPPORTED_EVALUATOR_TASKS = {
    "text-classification": {
        "implementation": TextClassificationEvaluator,
        "default_metric_name": "f1",
    }
}


def get_supported_tasks() -> List[str]:
    return SUPPORTED_EVALUATOR_TASKS.keys()


def check_task(task: str) -> Dict:
    """
    Checks an incoming task string, to validate it's correct and return the default Evaluator class metric name.
    """
    if task in TASK_ALIASES:
        task = TASK_ALIASES[task]
    if not check_pipeline_task(task):
        raise KeyError(f"Unknown task {task}, available tasks are: {get_supported_tasks()}.")
    if task in SUPPORTED_EVALUATOR_TASKS.keys() and task in SUPPORTED_PIPELINE_TASKS.keys():
        return SUPPORTED_EVALUATOR_TASKS[task]
    raise KeyError(f"Unknown task {task}, available tasks are: {get_supported_tasks()}.")


def evaluator(task: str = None) -> Evaluator:
    targeted_task = check_task(task)
    evaluator_class = targeted_task["implementation"]
    default_metric_name = targeted_task["default_metric_name"]
    return evaluator_class(task=task, default_metric_name=default_metric_name)

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


try:
    from scipy.stats import bootstrap

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from transformers import Pipeline, PreTrainedModel, PreTrainedTokenizer, TFPreTrainedModel, pipeline
    from transformers.pipelines import SUPPORTED_TASKS as SUPPORTED_PIPELINE_TASKS
    from transformers.pipelines import TASK_ALIASES
    from transformers.pipelines import check_task as check_pipeline_task

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from typing_extensions import Literal

from .loading import load
from .module import EvaluationModule
from .utils.logging import get_logger


logger = get_logger(__name__)


class Evaluator(ABC):
    """
    The Evaluator class is the class from which all evaluators inherit. Refer to this class for methods shared across
    different evaluators.

    Base class implementing evaluator operations.
    """

    def __init__(self, task: str, default_metric_name: str = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "If you want to use the `Evaluator` you need `transformers`. Run `pip install evaluate[evaluator]`."
            )
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "If you want to use the `Evaluator` you need `scipy>=1.7.1`. Run `pip install evaluate[evaluator]`."
            )
        self.task = task
        self.default_metric_name = default_metric_name

    @abstractmethod
    def _compute_predictions(self, pipe: "Pipeline", inputs, **predictions_parameters: Dict):
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
        """
        A utility function enabling the confidence interval calculation for metrics computed
        by the evaluator based on `scipy`'s `bootstrap` method.
        """
        bootstrap_dict = {}
        for key in metric_keys:
            bs = bootstrap(
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
            bootstrap_dict[key] = {
                "confidence_interval": (bs.confidence_interval.low, bs.confidence_interval.high),
                "standard_error": bs.standard_error,
            }
        return bootstrap_dict

    @abstractmethod
    def compute(
        self,
        model_or_pipeline: Union[str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"] = None,
        data: Union[str, Dataset] = None,
        metric: Union[str, EvaluationModule] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        **compute_parameters: Dict,
    ):
        """
        A core method of the `Evaluator` class, computes the metric value for a pipeline and dataset compatible
        with the task specified by the `Evaluator`.
        """
        raise NotImplementedError()


class TextClassificationEvaluator(Evaluator):
    """
    Text classification evaluator.

    This text classification evaluator can currently be loaded from [`evaluator`] using the default task name
    `text-classification` or with a `"sentiment-analysis"` alias.

    Methods in this class assume a data format compatible with the [`TextClassificationPipeline`] - a single textual
    feature as input and a categorical label as output.
    """

    def __init__(self, task="text-classification", default_metric_name=None):
        super().__init__(task, default_metric_name=default_metric_name)

    def _compute_predictions(self, pipe: "Pipeline", inputs, label_mapping: Dict[str, Number] = None) -> List[Number]:
        predictions = pipe(inputs, truncation=True)
        return [
            label_mapping[element["label"]] if label_mapping is not None else element["label"]
            for element in predictions
        ]

    def compute(
        self,
        model_or_pipeline: Union[str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"] = None,
        data: Union[str, Dataset] = None,
        metric: Union[str, EvaluationModule] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        random_state: Optional[int] = None,
        input_column: str = "text",
        label_column: str = "label",
        label_mapping: Optional[Dict[str, Number]] = None,
    ) -> Tuple[Dict[str, float], Any]:
        """
        Compute the metric for a given pipeline and dataset combination.

        Args:
            model_or_pipeline (`str` or `Pipeline` or `Callable` or `PreTrainedModel` or `TFPreTrainedModel`,
            defaults to `None`):
                If the argument in not specified, we initialize the default pipeline for the task (in this case
                `text-classification` or its alias - `sentiment-analysis`). If the argument is of the type `str` or
                is a model instance, we use it to initialize a new `Pipeline` with the given model. Otherwise we assume the
                argument specifies a pre-initialized pipeline.
            data (`str` or `Dataset`, defaults to `None):
                Specifies the dataset we will run evaluation on. If it is of type `str`, we treat it as the dataset
                name, and load it. Otherwise we assume it represents a pre-loaded dataset.
            metric (`str` or `EvaluationModule`, defaults to `None`"
                Specifies the metric we use in evaluator. If it is of type `str`, we treat it as the metric name, and
                load it. Otherwise we assume it represents a pre-loaded metric.
            tokenizer: (`str` or `PreTrainedTokenizer`, *optional*, defaults to `None`):
                Argument can be used to overwrite a default tokenizer if `model_or_pipeline` represents a model for
                which we build a pipeline. If `model_or_pipeline` is `None` or a pre-initialized pipeline, we ignore
                this argument.
            strategy: (`Literal["simple", "bootstrap"]`, defaults to "simple"):
                specifies the evaluation strategy. Possible values are:

                - `"simple"` - we evaluate the metric and return the scores.
                - `"bootstrap"` - on top of computing the metric scores, we calculate the confidence interval for each
                of the returned metric keys, using `scipy`'s `bootstrap` method
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html.

            confidence_level (`float`, defaults to `0.95`):
                The `confidence_level` value passed to `bootstrap` if `"bootstrap"` strategy is chosen.
            n_resamples (`int`, defaults to `9999`):
                The `n_resamples` value passed to `bootstrap` if `"bootstrap"` strategy is chosen.
            random_state (`int`, *optional*, defaults to `None`):
                The `random_state` value passed to `bootstrap` if `"bootstrap"` strategy is chosen. Useful for
                debugging.
            input_column (`str`, defaults to `"text"`):
                the name of the column containing the text feature in the dataset specified by `data`.
            label_column (`str`, defaults to `"label"`):
                the name of the column containing the labels in the dataset specified by `data`.
            label_mapping (`Dict[str, Number]`, *optional*, defaults to `None`):
                We want to map class labels defined by the model in the pipeline to values consistent with those
                defined in the `label_column` of the `data` dataset.

        Return:
            A `Dict`. The keys represent metric keys calculated for the `metric` spefied in function arguments. For the
            `"simple"` strategy, the value is the metric score. For the `"bootstrap"` strategy, the value is a `Dict`
            containing the score, the confidence interval and the standard error calculated for each metric key.

        Examples:

        ```python
        >>> from evaluate import evaluator
        >>> from datasets import Dataset, load_dataset

        >>> e = evaluator("text-classification")
        >>> data =  Dataset.from_dict(load_dataset("imdb")["test"][:2])

        >>> results = e.compute(
        >>>     model_or_pipeline="huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli",
        >>>     data=data,
        >>>     metric="accuracy",
        >>>     input_column="text",
        >>>     label_column="label",
        >>>     label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
        >>>     strategy="bootstrap",
        >>>     n_resamples=10,
        >>>     random_state=0
        >>> )
        ```"""
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
                    "`Evaluator` doesn't specify a default metric. Please specify a valid `metric` argument."
                )
            metric = load(self.default_metric_name)
        elif isinstance(metric, str):
            metric = load(metric)

        # Core computations.
        references = data[label_column]
        predictions = self._compute_predictions(pipe, data[input_column], label_mapping=label_mapping)
        result = metric.compute(predictions=predictions, references=references)

        if strategy == "bootstrap":
            metric_keys = result.keys()
            bootstrap_dict = Evaluator._compute_confidence_interval(
                predictions,
                references,
                metric,
                metric_keys,
                confidence_level,
                n_resamples,
                random_state,
            )
            for key in metric_keys:
                bootstrap_dict[key]["score"] = result[key]

            return bootstrap_dict

        return result


SUPPORTED_EVALUATOR_TASKS = {
    "text-classification": {
        "implementation": TextClassificationEvaluator,
        "default_metric_name": "accuracy",
    }
}


def get_supported_tasks() -> List[str]:
    """
    Returns a list of supported task strings.
    """
    return SUPPORTED_EVALUATOR_TASKS.keys()


def check_task(task: str) -> Dict:
    """
    Checks an incoming task string, to validate it's correct and returns the default Evaluator class and default metric
    name. It first performs a check to validata that the string is a valid `Pipeline` task, then it checks if it's a
    valid `Evaluator` task. `Evaluator` tasks are a substet of `Pipeline` tasks.

    Args:
        task (`str`):
            The task defining which evaluator will be returned. Currently accepted tasks are:

            - `"text-classification"` (alias `"sentiment-analysis"` available)

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

    Evaluators encapsulate a task and a default metric name. They leverate `pipeline` functionalify from `transformers`
    to simplify the evaluation of multiple combinations of models, datasets and metrics for a given task.

    Args:
        task (`str`):
            The task defining which evaluator will be returned. Currently accepted tasks are:

            - `"text-classification"` (alias `"sentiment-analysis"` available): will return a [`TextClassificationEvaluator`].

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

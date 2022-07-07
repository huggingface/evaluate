# Copyright 2022 The HuggingFace Evaluate Authors.
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

from typing import Any, Callable, Dict, Optional, Tuple, Union

# Lint as: python3
from datasets import Dataset, load_dataset


try:
    from transformers import Pipeline, PreTrainedModel, PreTrainedTokenizer, TFPreTrainedModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from typing_extensions import Literal

from ..module import EvaluationModule
from ..utils.logging import get_logger
from .base import Evaluator


logger = get_logger(__name__)


class QuestionAnsweringEvaluator(Evaluator):
    """
    Question answering evaluator.
    This question answering evaluator can currently be loaded from [`evaluator`] using the default task name
    `question-answering`.
    Methods in this class assume a data format compatible with the [`QuestionAnsweringPipeline`].
    """

    def __init__(self, task="question-answering", default_metric_name=None):
        super().__init__(task, default_metric_name=default_metric_name)

    def prepare_data(
        self,
        data: Union[str, Dataset],
        question_column: str,
        context_column: str,
        id_column: str,
        label_column: str,
    ):
        """Prepare data."""
        if data is None:
            raise ValueError(
                "Please specify a valid `data` object - either a `str` with a name or a `Dataset` object."
            )
        data = load_dataset(data) if isinstance(data, str) else data
        if question_column not in data.column_names:
            raise ValueError(
                f"Invalid `question_column` {question_column} specified. The dataset contains the following columns: {data.column_names}."
            )
        if context_column not in data.column_names:
            raise ValueError(
                f"Invalid `context_column` {context_column} specified. The dataset contains the following columns: {data.column_names}."
            )
        if id_column not in data.column_names:
            raise ValueError(
                f"Invalid `id_column` {id_column} specified. The dataset contains the following columns: {data.column_names}."
            )
        if label_column not in data.column_names:
            raise ValueError(
                f"Invalid `label_column` {label_column} specified. The dataset contains the following columns: {data.column_names}."
            )

        return data

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
        question_column: str = "question",
        context_column: str = "context",
        id_column: str = "id",
        label_column: str = "answers",
    ) -> Tuple[Dict[str, float], Any]:
        """
        Compute the metric for a given pipeline and dataset combination.
        Args:
            model_or_pipeline (`str` or `Pipeline` or `Callable` or `PreTrainedModel` or `TFPreTrainedModel`,
            defaults to `None`):
                If the argument in not specified, we initialize the default pipeline for the task (in this case
                `question-answering`). If the argument is of the type `str` or
                is a model instance, we use it to initialize a new `Pipeline` with the given model. Otherwise we assume the
                argument specifies a pre-initialized pipeline.
            data (`str` or `Dataset`, defaults to `None):
                Specifies the dataset we will run evaluation on. If it is of type `str`, we treat it as the dataset
                name, and load it. Otherwise we assume it represents a pre-loaded dataset.
            metric (`str` or `EvaluationModule`, defaults to `None`):"
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
            question_column (`str`, defaults to `"question"`):
                the name of the column containing the question in the dataset specified by `data`.
            context_column (`str`, defaults to `"context"`):
                the name of the column containing the context in the dataset specified by `data`.
            id_column (`str`, defaults to `"id"`):
                the name of the column cointaing the identification field of the question and answer pair in the
                dataset specified by `data`.
            label_column (`str`, defaults to `"answers"`):
                the name of the column containing the answers in the dataset specified by `data`.
        Return:
            A `Dict`. The keys represent metric keys calculated for the `metric` spefied in function arguments. For the
            `"simple"` strategy, the value is the metric score. For the `"bootstrap"` strategy, the value is a `Dict`
            containing the score, the confidence interval and the standard error calculated for each metric key.
        Examples:
        ```python
        >>> from evaluate import evaluator
        >>> from datasets import Dataset, load_dataset
        >>> e = evaluator("question-answering")
        >>> data = load_dataset("squad", split="validation[:2]")
        >>> results = e.compute(
        >>>     model_or_pipeline="mrm8488/bert-tiny-finetuned-squadv2",
        >>>     data=data,
        >>>     metric="squad",
        >>>     question_column="question",
        >>>     context_column="context",
        >>>     label_column="answers",
        >>>     strategy="bootstrap",
        >>>     n_resamples=10,
        >>>     random_state=0
        >>> )
        ```"""
        data = self.prepare_data(
            data=data,
            question_column=question_column,
            context_column=context_column,
            id_column=id_column,
            label_column=label_column,
        )

        pipe = self.prepare_pipeline(model_or_pipeline=model_or_pipeline, preprocessor=tokenizer)

        metric = self.prepare_metric(metric)

        # Compute predictions
        predictions = pipe(question=data[question_column], context=data[context_column], padding="max_length")
        predictions = [
            {"prediction_text": predictions[i]["answer"], "id": data[i][id_column]} for i in range(len(predictions))
        ]

        references = [{"id": element[id_column], "answers": element[label_column]} for element in data]

        # Compute metrics from references and predictions
        result = self.core_compute(
            references=references,
            predictions=predictions,
            metric=metric,
            strategy=strategy,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )

        return result

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

import hashlib
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dill
import pyarrow as pa
import pyarrow.parquet as pq

# Lint as: python3
from datasets import Dataset

from evaluate.utils import canary


try:
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from typing_extensions import Literal

from ..module import EvaluationModule
from ..utils.file_utils import add_end_docstrings, add_start_docstrings
from ..utils.logging import get_logger
from .base import EVALUATOR_COMPUTE_RETURN_DOCSTRING, EVALUTOR_COMPUTE_START_DOCSTRING, Evaluator
from .utils import DatasetColumn


logger = get_logger(__name__)


TASK_DOCUMENTATION = r"""
    Examples:
    ```python
    >>> from evaluate import evaluator
    >>> from datasets import load_dataset
    >>> task_evaluator = evaluator("question-answering")
    >>> data = load_dataset("squad", split="validation[:2]")
    >>> results = task_evaluator.compute(
    >>>     model_or_pipeline="sshleifer/tiny-distilbert-base-cased-distilled-squad",
    >>>     data=data,
    >>>     metric="squad",
    >>> )
    ```

    <Tip>

    Datasets where the answer may be missing in the context are supported, for example SQuAD v2 dataset. In this case, it is safer to pass `squad_v2_format=True` to
    the compute() call.

    </Tip>

    ```python
    >>> from evaluate import evaluator
    >>> from datasets import load_dataset
    >>> task_evaluator = evaluator("question-answering")
    >>> data = load_dataset("squad_v2", split="validation[:2]")
    >>> results = task_evaluator.compute(
    >>>     model_or_pipeline="mrm8488/bert-tiny-finetuned-squadv2",
    >>>     data=data,
    >>>     metric="squad_v2",
    >>>     squad_v2_format=True,
    >>> )
    ```
"""


class QuestionAnsweringEvaluator(Evaluator):
    """
    Question answering evaluator. This evaluator handles
    [**extractive** question answering](https://huggingface.co/docs/transformers/task_summary#extractive-question-answering),
    where the answer to the question is extracted from a context.

    This question answering evaluator can currently be loaded from [`evaluator`] using the default task name
    `question-answering`.

    Methods in this class assume a data format compatible with the
    [`QuestionAnsweringPipeline`](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline).
    """

    PIPELINE_KWARGS = {}

    def __init__(self, task="question-answering", default_metric_name=None):
        super().__init__(task, default_metric_name=default_metric_name)

    def prepare_data(
        self, data: Dataset, question_column: str, context_column: str, id_column: str, label_column: str
    ):
        """Prepare data."""
        if data is None:
            raise ValueError(
                "Please specify a valid `data` object - either a `str` with a name or a `Dataset` object."
            )
        self.check_required_columns(
            data,
            {
                "question_column": question_column,
                "context_column": context_column,
                "id_column": id_column,
                "label_column": label_column,
            },
        )

        metric_inputs = dict()
        metric_inputs["references"] = [
            {"id": element[id_column], "answers": element[label_column]} for element in data
        ]

        return metric_inputs, {
            "question": DatasetColumn(data, question_column),
            "context": DatasetColumn(data, context_column),
        }

    def is_squad_v2_format(self, data: Dataset, label_column: str = "answers"):
        """
        Check if the provided dataset follows the squad v2 data schema, namely possible samples where the answer is not in the context.
        In this case, the answer text list should be `[]`.
        """
        original_num_rows = data.num_rows
        nonempty_num_rows = data.filter(
            lambda x: len(x[label_column]["text"]) > 0, load_from_cache_file=False
        ).num_rows
        if original_num_rows > nonempty_num_rows:
            return True
        else:
            return False

    def predictions_processor(self, predictions: List, squad_v2_format: bool, ids: List):
        result = []
        for i in range(len(predictions)):
            pred = {"prediction_text": predictions[i]["answer"], "id": ids[i]}
            if squad_v2_format:
                pred["no_answer_probability"] = predictions[i]["score"]
            result.append(pred)
        return {"predictions": result}

    def compute_canary_hash(self, pipe, input_columns, label_column):
        canary_data = canary.CanaryDataset(self.task, input_columns, label_column)
        _, canary_inputs = self.prepare_data(
            data=canary_data.data,
            question_column=input_columns[0],
            context_column=input_columns[1],
            id_column=input_columns[0],
            label_column=label_column,
        )
        predictions, _ = self.call_pipeline(pipe, canary_inputs)
        return hashlib.md5(dill.dumps(predictions)).hexdigest()

    @add_start_docstrings(EVALUTOR_COMPUTE_START_DOCSTRING)
    @add_end_docstrings(EVALUATOR_COMPUTE_RETURN_DOCSTRING, TASK_DOCUMENTATION)
    def compute(
        self,
        model_or_pipeline: Union[
            str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"  # noqa: F821
        ] = None,
        data: Union[str, Dataset] = None,
        split: Optional[str] = None,
        metric: Union[str, EvaluationModule] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,  # noqa: F821
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        device: int = None,
        random_state: Optional[int] = None,
        question_column: str = "question",
        context_column: str = "context",
        id_column: str = "id",
        label_column: str = "answers",
        squad_v2_format: Optional[bool] = None,
        cache_if_possible=False,
    ) -> Tuple[Dict[str, float], Any]:
        """
        question_column (`str`, defaults to `"question"`):
            the name of the column containing the question in the dataset specified by `data`.
        context_column (`str`, defaults to `"context"`):
            the name of the column containing the context in the dataset specified by `data`.
        id_column (`str`, defaults to `"id"`):
            the name of the column containing the identification field of the question and answer pair in the
            dataset specified by `data`.
        label_column (`str`, defaults to `"answers"`):
            the name of the column containing the answers in the dataset specified by `data`.
        squad_v2_format (`bool`, *optional*, defaults to `None`):
            whether the dataset follows the format of squad_v2 dataset. This is the case when the provided dataset
            has questions where the answer is not in the context, more specifically when are answers as
            `{"text": [], "answer_start": []}` in the answer column. If all questions have at least one answer, this parameter
            should be set to `False`. If this parameter is not provided, the format will be automatically inferred.
        """
        result = {}

        data = self.load_data(data=data, split=split)
        metric_inputs, pipe_inputs = self.prepare_data(
            data=data,
            question_column=question_column,
            context_column=context_column,
            id_column=id_column,
            label_column=label_column,
        )

        if squad_v2_format is None:
            squad_v2_format = self.is_squad_v2_format(data=data, label_column=label_column)
            logger.warning(
                f"`squad_v2_format` parameter not provided to QuestionAnsweringEvaluator.compute(). Automatically inferred `squad_v2_format` as {squad_v2_format}."
            )
        pipe = self.prepare_pipeline(model_or_pipeline=model_or_pipeline, tokenizer=tokenizer, device=device)

        metric = self.prepare_metric(metric)

        if squad_v2_format and metric.name == "squad":
            logger.warning(
                "The dataset has SQuAD v2 format but you are using the SQuAD metric. Consider passing the 'squad_v2' metric."
            )
        if not squad_v2_format and metric.name == "squad_v2":
            logger.warning(
                "The dataset has SQuAD v1 format but you are using the SQuAD v2 metric. Consider passing the 'squad' metric."
            )

        if squad_v2_format:
            self.PIPELINE_KWARGS["handle_impossible_answer"] = True
        else:
            self.PIPELINE_KWARGS["handle_impossible_answer"] = False

        # If `cache_if_possible` = True, test whether this exact pipe has been instantiated before
        if cache_if_possible:
            input_columns = [question_column, context_column, id_column]
            canary_hash = self.compute_canary_hash(pipe, input_columns, label_column)

            # Check if model, data, metric combination has already been computed and cached
            cache_file_name = os.path.join(
                self.cache_dir, f"cache-{canary_hash}-{data._fingerprint}-{metric._hash}" + ".parquet"
            )

            # Retrieve computed results from the cache if they already exist
            if os.path.exists(cache_file_name):
                logger.warning(f"Loading cached computed results at {cache_file_name}")
                result_from_table = pa.Table.to_pydict(pq.read_table(cache_file_name))
                result = {k: v[0] for (k, v) in result_from_table.items()}
                return result

        # Compute predictions
        predictions, perf_results = self.call_pipeline(pipe, **pipe_inputs)
        predictions = self.predictions_processor(predictions, squad_v2_format=squad_v2_format, ids=data[id_column])
        metric_inputs.update(predictions)

        # Compute metrics from references and predictions
        metric_results = self.compute_metric(
            metric=metric,
            metric_inputs=metric_inputs,
            strategy=strategy,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )

        result.update(metric_results)
        result.update(perf_results)

        # Cache evaluation results.
        #   These can be removed by calling evaluate.utils.file_utils.cleanup_cache_files(self.cache_dir)
        if cache_if_possible and canary_hash:
            self.write_to_cache(cache_file_name, result)

        return result

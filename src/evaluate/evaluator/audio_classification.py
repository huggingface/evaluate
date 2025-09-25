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

from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

from datasets import Dataset
from typing_extensions import Literal

from ..module import EvaluationModule
from ..utils.file_utils import add_end_docstrings, add_start_docstrings
from .base import EVALUATOR_COMPUTE_RETURN_DOCSTRING, EVALUTOR_COMPUTE_START_DOCSTRING, Evaluator


if TYPE_CHECKING:
    from transformers import FeatureExtractionMixin, Pipeline, PreTrainedModel, PreTrainedTokenizer, TFPreTrainedModel


TASK_DOCUMENTATION = r"""
    Examples:

    > [!TIP]
    > Remember that, in order to process audio files, you need ffmpeg installed (https://ffmpeg.org/download.html)

    ```python
    >>> from evaluate import evaluator
    >>> from datasets import load_dataset

    >>> task_evaluator = evaluator("audio-classification")
    >>> data = load_dataset("superb", 'ks', split="test[:40]")
    >>> results = task_evaluator.compute(
    >>>     model_or_pipeline=""superb/wav2vec2-base-superb-ks"",
    >>>     data=data,
    >>>     label_column="label",
    >>>     input_column="file",
    >>>     metric="accuracy",
    >>>     label_mapping={0: "yes", 1: "no", 2: "up", 3: "down"}
    >>> )
    ```

    > [!TIP]
    > The evaluator supports raw audio data as well, in the form of a numpy array. However, be aware that calling
    > the audio column automatically decodes and resamples the audio files, which can be slow for large datasets.

    ```python
    >>> from evaluate import evaluator
    >>> from datasets import load_dataset

    >>> task_evaluator = evaluator("audio-classification")
    >>> data = load_dataset("superb", 'ks', split="test[:40]")
    >>> data = data.map(lambda example: {"audio": example["audio"]["array"]})
    >>> results = task_evaluator.compute(
    >>>     model_or_pipeline=""superb/wav2vec2-base-superb-ks"",
    >>>     data=data,
    >>>     label_column="label",
    >>>     input_column="audio",
    >>>     metric="accuracy",
    >>>     label_mapping={0: "yes", 1: "no", 2: "up", 3: "down"}
    >>> )
    ```
"""


class AudioClassificationEvaluator(Evaluator):
    """
    Audio classification evaluator.
    This audio classification evaluator can currently be loaded from [`evaluator`] using the default task name
    `audio-classification`.
    Methods in this class assume a data format compatible with the [`transformers.AudioClassificationPipeline`].
    """

    PIPELINE_KWARGS = {}

    def __init__(self, task="audio-classification", default_metric_name=None):
        super().__init__(task, default_metric_name=default_metric_name)

    def predictions_processor(self, predictions, label_mapping):
        pred_label = [max(pred, key=lambda x: x["score"])["label"] for pred in predictions]
        pred_label = [label_mapping[pred] if label_mapping is not None else pred for pred in pred_label]

        return {"predictions": pred_label}

    @add_start_docstrings(EVALUTOR_COMPUTE_START_DOCSTRING)
    @add_end_docstrings(EVALUATOR_COMPUTE_RETURN_DOCSTRING, TASK_DOCUMENTATION)
    def compute(
        self,
        model_or_pipeline: Union[
            str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"  # noqa: F821
        ] = None,
        data: Union[str, Dataset] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        metric: Union[str, EvaluationModule] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,  # noqa: F821
        feature_extractor: Optional[Union[str, "FeatureExtractionMixin"]] = None,  # noqa: F821
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        device: int = None,
        random_state: Optional[int] = None,
        input_column: str = "file",
        label_column: str = "label",
        label_mapping: Optional[Dict[str, Number]] = None,
    ) -> Tuple[Dict[str, float], Any]:

        """
        input_column (`str`, defaults to `"file"`):
            The name of the column containing either the audio files or a raw waveform, represented as a numpy array, in the dataset specified by `data`.
        label_column (`str`, defaults to `"label"`):
            The name of the column containing the labels in the dataset specified by `data`.
        label_mapping (`Dict[str, Number]`, *optional*, defaults to `None`):
            We want to map class labels defined by the model in the pipeline to values consistent with those
            defined in the `label_column` of the `data` dataset.
        """

        result = super().compute(
            model_or_pipeline=model_or_pipeline,
            data=data,
            subset=subset,
            split=split,
            metric=metric,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            strategy=strategy,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            device=device,
            random_state=random_state,
            input_column=input_column,
            label_column=label_column,
            label_mapping=label_mapping,
        )

        return result

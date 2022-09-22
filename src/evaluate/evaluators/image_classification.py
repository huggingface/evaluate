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
from typing import Any, Callable, Dict, Optional, Tuple, Union

from datasets import Dataset
from typing_extensions import Literal

from ..module import EvaluationModule
from ..utils.file_utils import add_end_docstrings, add_start_docstrings
from .base import EVALUATOR_COMPUTE_RETURN_DOCSTRING, EVALUTOR_COMPUTE_START_DOCSTRING, Evaluator


TASK_DOCUMENTATION = r"""
    Examples:
    ```python
    >>> from evaluate import evaluator
    >>> from datasets import load_dataset
    >>> task_evaluator = evaluator("image-classification")
    >>> data = load_dataset("beans", split="test[:40]")
    >>> results = task_evaluator.compute(
    >>>     model_or_pipeline="nateraw/vit-base-beans",
    >>>     data=data,
    >>>     label_column="labels",
    >>>     metric="accuracy",
    >>>     label_mapping={'angular_leaf_spot': 0, 'bean_rust': 1, 'healthy': 2},
    >>>     strategy="bootstrap"
    >>> )
    ```
"""


class ImageClassificationEvaluator(Evaluator):
    """
    Image classification evaluator.
    This image classification evaluator can currently be loaded from [`evaluator`] using the default task name
    `image-classification`.
    Methods in this class assume a data format compatible with the [`ImageClassificationPipeline`].
    """

    PIPELINE_KWARGS = {}

    def __init__(self, task="image-classification", default_metric_name=None):
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
        metric: Union[str, EvaluationModule] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,  # noqa: F821
        feature_extractor: Optional[Union[str, "FeatureExtractionMixin"]] = None,  # noqa: F821
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        device: int = None,
        random_state: Optional[int] = None,
        input_column: str = "image",
        label_column: str = "label",
        label_mapping: Optional[Dict[str, Number]] = None,
    ) -> Tuple[Dict[str, float], Any]:

        """
        input_column (`str`, defaults to `"image"`):
            the name of the column containing the images as PIL ImageFile in the dataset specified by `data`.
        label_column (`str`, defaults to `"label"`):
            the name of the column containing the labels in the dataset specified by `data`.
        label_mapping (`Dict[str, Number]`, *optional*, defaults to `None`):
            We want to map class labels defined by the model in the pipeline to values consistent with those
            defined in the `label_column` of the `data` dataset.
        """

        result = super().compute(
            model_or_pipeline=model_or_pipeline,
            data=data,
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

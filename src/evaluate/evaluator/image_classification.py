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

from numbers import Number
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Lint as: python3
from datasets import Dataset


try:
    from transformers import FeatureExtractionMixin, Pipeline, PreTrainedModel, TFPreTrainedModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from typing_extensions import Literal

from ..module import EvaluationModule
from ..utils.logging import get_logger
from .base import Evaluator


logger = get_logger(__name__)


class ImageClassificationEvaluator(Evaluator):
    """
    Image classification evaluator.
    This image classification evaluator can currently be loaded from [`evaluator`] using the default task name
    `image-classification`.
    Methods in this class assume a data format compatible with the [`ImageClassificationPipeline`].
    """

    def __init__(self, task="image-classification", default_metric_name=None):
        super().__init__(task, default_metric_name=default_metric_name)

    def predictions_processor(self, predictions, label_mapping):
        pred_label = [max(pred, key=lambda x: x["score"])["label"] for pred in predictions]
        return [label_mapping[pred] if label_mapping is not None else pred for pred in pred_label]

    def compute(
        self,
        model_or_pipeline: Union[str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"] = None,
        data: Union[str, Dataset] = None,
        metric: Union[str, EvaluationModule] = None,
        feature_extractor: Optional[Union[str, "FeatureExtractionMixin"]] = None,
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        random_state: Optional[int] = None,
        input_column: str = "image",
        label_column: str = "labels",
        label_mapping: Optional[Dict[str, Number]] = None,
    ) -> Tuple[Dict[str, float], Any]:
        """
        Compute the metric for a given pipeline and dataset combination.
        Args:
            model_or_pipeline (`str` or `Pipeline` or `Callable` or `PreTrainedModel` or `TFPreTrainedModel`,
            defaults to `None`):
                If the argument in not specified, we initialize the default pipeline for the task (in this case
                `image-classification`. If the argument is of the type `str` or
                is a model instance, we use it to initialize a new `Pipeline` with the given model. Otherwise we assume the
                argument specifies a pre-initialized pipeline.
            data (`str` or `Dataset`, defaults to `None):
                Specifies the dataset we will run evaluation on. If it is of type `str`, we treat it as the dataset
                name, and load it. Otherwise we assume it represents a pre-loaded dataset.
            metric (`str` or `EvaluationModule`, defaults to `None`"
                Specifies the metric we use in evaluator. If it is of type `str`, we treat it as the metric name, and
                load it. Otherwise we assume it represents a pre-loaded metric.
            feature_extractor: (`str` or `FeatureExtractionMixin`, *optional*, defaults to `None`):
                Argument can be used to overwrite a default feature extractor if `model_or_pipeline` represents a model for
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
            input_column (`str`, defaults to `"image"`):
                the name of the column containing the images as PIL ImageFile in the dataset specified by `data`.
            label_column (`str`, defaults to `"labels"`):
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
        >>> e = evaluator("image-classification")
        >>> data =  Dataset.from_dict(load_dataset("beans")["test"][:2])
        >>> results = e.compute(
        >>>     model_or_pipeline="nateraw/vit-base-beans",
        >>>     data=data,
        >>>     metric="accuracy",
        >>>     input_column="image",
        >>>     label_column="labels",
        >>>     label_mapping={'angular_leaf_spot': 0, 'bean_rust': 1, 'healthy': 2},
        >>>     strategy="bootstrap",
        >>>     n_resamples=10,
        >>>     random_state=0
        >>> )
        ```"""

        result= {}
        
        data = self.prepare_data(data=data, input_column=input_column, label_column=label_column)
        metric_inputs = {"references": data[label_column]}

        pipe = self.prepare_pipeline(model_or_pipeline=model_or_pipeline, preprocessor=feature_extractor)

        metric = self.prepare_metric(metric)

        # Compute predictions
        predictions = self.call_pipeline(pipe, data[input_column])
        metric_inputs["predictions"] = self.predictions_processor(predictions, label_mapping)
        
        # Compute metrics from references and predictions
        result.update(self.compute_metric(
            metric=metric,
            metric_inputs=metric_inputs,
            strategy=strategy,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
            )
        )

        return result

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
from datasets import Dataset, load_dataset, load_metric
from datasets.metric import Metric
from scipy.stats import bootstrap
from transformers import Pipeline, pipeline
from typing_extensions import Literal


class Evaluator(ABC):
    def __init__(self, task, default_metric=None):
        self.task = task
        self.default_metric = default_metric

    @abstractmethod
    def _compute_predictions(self, pipe: Pipeline, inputs, **predictions_parameters: Dict):
        raise NotImplementedError()

    @staticmethod
    def _compute_confidence_interval(
        predictions,
        references,
        metric: Metric,
        metric_keys: List[str],
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
    ) -> Dict[str, Any]:
        bootstrap_dict = {}
        for key in metric_keys:
            bootstrap_dict[key] = bootstrap(
                data=(predictions, references),
                statistic=lambda predictions, references: metric.compute(
                    predictions=predictions, references=references
                )[key],
                paired=True,
                vectorized=False,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
            )
        return bootstrap_dict

    @abstractmethod
    def compute(
        self,
        model: str = None,
        pipe: Union[Pipeline, Callable] = None,
        data: Union[str, Dataset] = None,
        metric: Union[str, Metric] = None,
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        **compute_parameters: Dict,
    ):
        raise NotImplementedError()


class TextClassificationEvaluator(Evaluator):
    def __init__(self, task="text-classification", default_metric=None):
        super().__init__(task, default_metric=default_metric)

    def _compute_predictions(self, pipe: Pipeline, inputs, label_mapping: Dict[str, Number] = None) -> List[Number]:
        predictions = pipe(inputs, truncation=True)
        return [
            label_mapping[element["label"]] if label_mapping is not None else element["label"]
            for element in predictions
        ]

    def compute(
        self,
        model: str = None,
        pipe: Union[Pipeline, Callable] = None,
        data: Union[str, Dataset] = None,
        metric: Union[str, Metric] = None,
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        input_column: str = "inputs",
        label_column: str = "references",
        label_mapping: Optional[Dict[str, Number]] = None,
    ) -> Tuple[Dict[str, float], Any]:
        assert data is not None

        pipe = (
            pipeline(self.task, model=model)
            if model is not None
            else pipe
            if pipe is not None
            else pipeline(self.task)
        )
        assert pipe.task == self.task

        metric = (
            load_metric(metric)
            if isinstance(metric, str)
            else (metric if metric is not None else (self.default_metric if self.default_metric is not None else None))
        )
        assert metric is not None

        data = load_dataset(data) if isinstance(data, str) else data

        assert (
            input_column in data.column_names
        ), f"Invalid `input_column` specified: the dataset contains the columns {data.column_names}."
        assert (
            label_column in data.column_names
        ), f"Invalid `label_column` specified: the dataset contains the columns {data.column_names}."

        references = data[label_column]
        predictions = self._compute_predictions(pipe, data[input_column], label_mapping=label_mapping)
        result = metric.compute(predictions=predictions, references=references)

        bootstrap = (
            Evaluator._compute_confidence_interval(
                predictions, references, metric, result.keys(), confidence_level, n_resamples
            )
            if strategy == "bootstrap"
            else None
        )
        return result, bootstrap

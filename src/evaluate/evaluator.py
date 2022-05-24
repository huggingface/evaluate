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
from typing import Dict, Optional, Union

# Lint as: python3
import numpy as np
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_metric
from datasets.metric import Metric
from scipy.stats import bootstrap
from transformers import Pipeline, pipeline


class Evaluator:
    def __init__(
        self,
        pipe: Union[str, Pipeline],
        data: Union[str, DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        metric: Union[str, Metric],
        metric_key: Optional[str] = None,
        col_mapping: Optional[Dict[str, str]] = None,
        label_mapping: Optional[Dict[str, Number]] = None,
    ):
        self.pipe = pipeline(pipe) if isinstance(pipe, str) else pipe
        self.data = load_dataset(data) if isinstance(data, str) else data
        self.metric = load_metric(metric) if isinstance(metric, str) else metric
        self.metric_key = metric_key
        self.label_mapping = label_mapping
        if col_mapping is not None:
            self.data = data.rename_columns(col_mapping)

    def get_predictions(
        self, baseline_pipe: Optional[Pipeline] = None, baseline_label_mapping: Optional[Dict[str, Number]] = None
    ):
        assert not (
            baseline_pipe is None and baseline_label_mapping is not None
        ), "You can only provide `baseline_label_mapping` if `baseline_pipe` is provided."
        predictions = (
            baseline_pipe(self.data["inputs"], truncation=True)
            if baseline_pipe is not None
            else self.pipe(self.data["inputs"], truncation=True)
        )
        label_mapping = self.label_mapping if baseline_label_mapping is None else baseline_label_mapping
        return [
            label_mapping[element["label"]] if label_mapping is not None else element["label"]
            for element in predictions
        ]

    def get_references(self):
        return self.data["references"]

    def statistic(self, predictions=None, references=None, metric_key: Optional[str] = None):
        return self.metric.compute(predictions=predictions, references=references)[
            self.metric_key if metric_key is None else metric_key
        ]

    def get_confidence_interval(self, confidence_level: float = 0.95, n_resamples: int = 9999):
        predictions = self.get_predictions()
        references = self.get_references()
        return bootstrap(data=(predictions, references), statistic=self.statistic, paired=True, vectorized=False)

    def get_bootstrap_p_value(
        self,
        baseline_pipe: Union[str, Pipeline],
        baseline_label_mapping: Optional[Dict[str, Number]] = None,
        n_resamples: int = 9999,
    ):
        """Compare two model's scores by getting p-values from non-parametric bootstrap. Here, the null hypothesis is:
        "current model is no better than the baseline model on the test population".
        The implementation is based on the paper: https://aclanthology.org/D12-1091.pdf.
        """
        baseline_pipe = pipeline(baseline_pipe) if isinstance(baseline_pipe, str) else baseline_pipe
        preds = np.array(self.get_predictions())
        baseline_preds = np.array(
            self.get_predictions(baseline_pipe=baseline_pipe, baseline_label_mapping=baseline_label_mapping)
        )
        references = np.array(self.get_references())

        delta = self.statistic(preds, references) - self.statistic(baseline_preds, references)
        s = 0
        for i in range(n_resamples):
            resampled_indices = np.random.choice(len(references), size=len(references), replace=True)
            delta_i = self.statistic(preds[resampled_indices], references[resampled_indices]) - self.statistic(
                baseline_preds[resampled_indices], references[resampled_indices]
            )
            if delta_i > 2 * delta:
                s += 1
        return s / n_resamples

    def compute(self):
        predictions = self.get_predictions()
        references = self.get_references()
        result = self.metric.compute(predictions=predictions, references=references)
        return result

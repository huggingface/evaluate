# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""MASE - Mean Absolute Scaled Error Metric"""

import datasets
import numpy as np
from sklearn.metrics import mean_absolute_error

import evaluate


_CITATION = """\
@article{HYNDMAN2006679,
    title = {Another look at measures of forecast accuracy},
    journal = {International Journal of Forecasting},
    volume = {22},
    number = {4},
    pages = {679--688},
    year = {2006},
    issn = {0169-2070},
    doi = {https://doi.org/10.1016/j.ijforecast.2006.03.001},
    url = {https://www.sciencedirect.com/science/article/pii/S0169207006000239},
    author = {Rob J. Hyndman and Anne B. Koehler},
}
"""

_DESCRIPTION = """\
Mean Absolute Scaled Error (MASE) is the mean absolute error of the forecast values, divided by the mean absolute error of the in-sample one-step naive forecast.
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    references: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    training: array-like of shape (n_samples,) or (n_samples, n_outputs)
        In sample training data for naive forecast.
    periodicity: int, default=1
        Seasonal periodicity of training data.
    sample_weight: array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput: {"raw_values", "uniform_average"} or array-like of shape (n_outputs,), default="uniform_average"
        Defines aggregating of multiple output values. Array-like value defines weights used to average errors.

                 "raw_values" : Returns a full set of errors in case of multioutput input.

                 "uniform_average" : Errors of all outputs are averaged with uniform weight.

Returns:
    mase : mean absolute scaled error.
        If multioutput is "raw_values", then mean absolute percentage error is returned for each output separately. If multioutput is "uniform_average" or an ndarray of weights, then the weighted average of all output errors is returned.
        MASE output is non-negative floating point. The best value is 0.0.
Examples:

    >>> mase_metric = evaluate.load("mase")
    >>> predictions = [2.5, 0.0, 2, 8, 1.25]
    >>> references = [3, -0.5, 2, 7, 2]
    >>> training = [5, 0.5, 4, 6, 3, 5, 2]
    >>> results = mase_metric.compute(predictions=predictions, references=references, training=training)
    >>> print(results)
    {'mase': 0.18333333333333335}

    If you're using multi-dimensional lists, then set the config as follows :

    >>> mase_metric = evaluate.load("mase", "multilist")
    >>> predictions = [[0, 2], [-1, 2], [8, -5]]
    >>> references = [[0.5, 1], [-1, 1], [7, -6]]
    >>> training = [[0.5, 1], [-1, 1], [7, -6]]
    >>> results = mase_metric.compute(predictions=predictions, references=references, training=training)
    >>> print(results)
    {'mase': 0.18181818181818182}
    >>> results = mase_metric.compute(predictions=predictions, references=references, training=training, multioutput='raw_values')
    >>> print(results)
    {'mase': array([0.10526316, 0.28571429])}
    >>> results = mase_metric.compute(predictions=predictions, references=references, training=training, multioutput=[0.3, 0.7])
    >>> print(results)
    {'mase': 0.21935483870967742}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Mase(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(self._get_feature_types()),
            reference_urls=["https://otexts.com/fpp2/accuracy.html#scaled-errors"],
        )

    def _get_feature_types(self):
        if self.config_name == "multilist":
            return {
                "predictions": datasets.Sequence(datasets.Value("float")),
                "references": datasets.Sequence(datasets.Value("float")),
            }
        else:
            return {
                "predictions": datasets.Value("float"),
                "references": datasets.Value("float"),
            }

    def _compute(
        self,
        predictions,
        references,
        training,
        periodicity=1,
        sample_weight=None,
        multioutput="uniform_average",
    ):

        y_pred_naive = training[:-periodicity]
        mae_naive = mean_absolute_error(training[periodicity:], y_pred_naive, multioutput=multioutput)

        mae_score = mean_absolute_error(
            references,
            predictions,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )

        epsilon = np.finfo(np.float64).eps
        mase_score = mae_score / np.maximum(mae_naive, epsilon)

        return {"mase": mase_score}

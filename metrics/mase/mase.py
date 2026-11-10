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
from sklearn.utils.validation import check_array

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
    training: array-like of shape (n_train_samples,) or (n_train_samples, n_outputs)
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
        Each output's forecast error is scaled by its own naive forecast error before aggregation.
        If multioutput is "raw_values", return each output's scaled error. Otherwise return their uniform or weighted average.
        MASE output is non-negative floating point. The best value is 0.0.
Examples:

    >>> mase_metric = evaluate.load("mase")
    >>> predictions = [2.5, 0.0, 2, 8, 1.25]
    >>> references = [3, -0.5, 2, 7, 2]
    >>> training = [5, 0.5, 4, 6, 3, 5, 2]
    >>> results = mase_metric.compute(predictions=predictions, references=references, training=training)
    >>> round(results["mase"], 6)
    0.183333

    If you're using multi-dimensional lists, then set the config as follows :

    >>> mase_metric = evaluate.load("mase", "multilist")
    >>> predictions = [[0, 2], [-1, 2], [8, -5]]
    >>> references = [[0.5, 1], [-1, 1], [7, -6]]
    >>> training = [[0.5, 1], [-1, 1], [7, -6]]
    >>> results = mase_metric.compute(predictions=predictions, references=references, training=training)
    >>> round(results["mase"], 6)
    0.195489
    >>> results = mase_metric.compute(predictions=predictions, references=references, training=training, multioutput='raw_values')
    >>> print(results)
    {'mase': array([0.10526316, 0.28571429])}
    >>> results = mase_metric.compute(predictions=predictions, references=references, training=training, multioutput=[0.3, 0.7])
    >>> round(results["mase"], 6)
    0.231579
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Mase(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(self._get_feature_types()),
            reference_urls=["https://otexts.com/fpp3/accuracy.html#scaled-errors"],
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
        mae_naive = mean_absolute_error(training[periodicity:], y_pred_naive, multioutput="raw_values")

        mae_score = mean_absolute_error(
            references,
            predictions,
            sample_weight=sample_weight,
            multioutput="raw_values",
        )

        if mae_score.shape != mae_naive.shape:
            raise ValueError("`training` and `references` must have the same number of outputs.")

        epsilon = np.finfo(np.float64).eps
        mase_score = mae_score / np.maximum(mae_naive, epsilon)

        if isinstance(multioutput, str):
            if multioutput == "raw_values":
                return {"mase": mase_score}
            if multioutput == "uniform_average":
                return {"mase": float(np.mean(mase_score))}
            raise ValueError('`multioutput` must be "raw_values", "uniform_average", or an array of weights.')

        weights = check_array(multioutput, ensure_2d=False)
        if mase_score.size == 1:
            raise ValueError("Custom weights are useful only in multi-output cases.")
        if weights.shape != mase_score.shape:
            raise ValueError("`multioutput` must provide one weight per output.")
        return {"mase": float(np.average(mase_score, weights=weights))}

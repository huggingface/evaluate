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
"""sMAPE - Symmetric Mean Absolute Percentage Error Metric"""

import datasets
import numpy as np
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils.validation import check_consistent_length

import evaluate


_CITATION = """\
@article{article,
    author = {Chen, Zhuo and Yang, Yuhong},
    year = {2004},
    month = {04},
    pages = {},
    title = {Assessing forecast accuracy measures}
}
"""

_DESCRIPTION = """\
Symmetric Mean Absolute Percentage Error (sMAPE) is the symmetric mean percentage error 
difference between the predicted and actual values as defined by Chen and Yang (2004),
based on the metric by Armstrong (1985) and Makridakis (1993).
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    references: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    sample_weight: array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput: {"raw_values", "uniform_average"} or array-like of shape (n_outputs,), default="uniform_average"
        Defines aggregating of multiple output values. Array-like value defines weights used to average errors.

                 "raw_values" : Returns a full set of errors in case of multioutput input.

                 "uniform_average" : Errors of all outputs are averaged with uniform weight.

Returns:
    smape : symmetric mean absolute percentage error.
        If multioutput is "raw_values", then symmetric mean absolute percentage error is returned for each output separately. If multioutput is "uniform_average" or an ndarray of weights, then the weighted average of all output errors is returned.
        sMAPE output is non-negative floating point in the range (0, 2). The best value is 0.0.
Examples:

    >>> smape_metric = evaluate.load("smape")
    >>> predictions = [2.5, 0.0, 2, 8]
    >>> references = [3, -0.5, 2, 7]
    >>> results = smape_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'smape': 0.5787878787878785}

    If you're using multi-dimensional lists, then set the config as follows :

    >>> smape_metric = evaluate.load("smape", "multilist")
    >>> predictions = [[0.5, 1], [-1, 1], [7, -6]]
    >>> references = [[0.1, 2], [-1, 2], [8, -5]]
    >>> results = smape_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'smape': 0.49696969558995985}
    >>> results = smape_metric.compute(predictions=predictions, references=references, multioutput='raw_values')
    >>> print(results)
    {'smape': array([0.48888889, 0.50505051])}
"""


def symmetric_mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    """Symmetric Mean absolute percentage error (sMAPE) metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    Returns
    -------
    loss : float or ndarray of floats
        If multioutput is 'raw_values', then mean absolute percentage error
        is returned for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.
        sMAPE output is non-negative floating point. The best value is 0.0.
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    smape = 2 * np.abs(y_pred - y_true) / (np.maximum(np.abs(y_true), epsilon) + np.maximum(np.abs(y_pred), epsilon))
    output_errors = np.average(smape, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Smape(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(self._get_feature_types()),
            reference_urls=["https://robjhyndman.com/hyndsight/smape/"],
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

    def _compute(self, predictions, references, sample_weight=None, multioutput="uniform_average"):

        smape_score = symmetric_mean_absolute_percentage_error(
            references,
            predictions,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )

        return {"smape": smape_score}

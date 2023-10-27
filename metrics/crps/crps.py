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
"""CRPS - Continuous Ranked Probability Score Metric"""

import datasets
import numpy as np

import evaluate


_CITATION = """\
@article{doi:10.1198/016214506000001437,
    author = {Tilmann Gneiting and Adrian E Raftery},
    title = {Strictly Proper Scoring Rules, Prediction, and Estimation},
    journal = {Journal of the American Statistical Association},
    volume = {102},
    number = {477},
    pages = {359--378},
    year = {2007},
    publisher = {Taylor & Francis},
    doi = {10.1198/016214506000001437},
    URL = {https://doi.org/10.1198/016214506000001437},
    eprint = {https://doi.org/10.1198/016214506000001437}
}
"""

_DESCRIPTION = """\
Continuous Ranked Probability Score (CRPS) is the generalization of mean absolute error to the case of probabilistic forecasts used to assess the respective accuracy of probabilistic forecasting methods.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: array-like of shape (n_samples,) or (n_samples, n_timesteps, n_outputs)
        n_samples from estimated target distribution.
    references: array-like of shape (1,) or (n_timesteps, n_outputs)
        Empirical (correct) target values from ground truth distribution.
    quantiles: list of floats, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        List of quantiles in the unit interval to compute CRPS over.
    sum: bool, default=False
        Defines whether to sum over sum_axis dimension.
    sum_axis: int, default=-1
        Defines axis to sum over in case of n_outputs > 1.
    multioutput: {"raw_values", "uniform_average"}
        Defines aggregating across the n_outputs dimension.
        "raw_values" returns full set of scores in case of multioutput input.
        "uniform_average" returns the average score across all outputs.
Returns:
    crps: float
        Continuous Ranked Probability Score.

Examples:

    >>> crps_metric = evaluate.load("crps")
    >>> predictions = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    >>> references = np.array([0.3])
    >>> results = crps_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'crps': 1.9999999254941967}

    >>> crps_metric = evaluate.load("crps", "multilist")
    >>> predictions = [[[0, 2, 4]], [[-1, 2, 5]], [[8, -5, 6]]]
    >>> references = [[0.5], [-1], [7]]
    >>> print(results)
    {'crps': 3.0522875816993467}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Crps(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(self._get_feature_types()),
            reference_urls=["https://www.lokad.com/continuous-ranked-probability-score/"],
        )

    def _get_feature_types(self):
        if self.config_name == "multilist":
            return {
                "predictions": datasets.Sequence(datasets.Sequence(datasets.Value("float"))),
                "references": datasets.Sequence(datasets.Value("float")),
            }
        else:
            return {
                "predictions": datasets.Sequence(datasets.Value("float")),
                "references": datasets.Value("float"),
            }

    @staticmethod
    def quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> float:
        return 2.0 * np.sum(np.abs((target - forecast) * ((target <= forecast) - q)))

    def _compute(
        self,
        predictions,
        references,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        sum=False,
        sum_axis=-1,
        multioutput="uniform_average",
    ):
        # if the number of dims of predictions > 2 then sum over sum_axis dimension if sum is True
        if sum and len(predictions.shape) > 1:
            predictions = np.sum(predictions, axis=sum_axis)
            references = np.sum(references, axis=sum_axis)

        abs_target_sum = np.sum(np.abs(references))
        weighted_quantile_loss = []
        for q in quantiles:
            forecast_quantile = np.quantile(predictions, q, axis=0)
            weighted_quantile_loss.append(self.quantile_loss(references, forecast_quantile, q) / abs_target_sum)

        if multioutput == "raw_values":
            return {"crps": weighted_quantile_loss}
        elif multioutput == "uniform_average":
            return {"crps": np.average(weighted_quantile_loss)}
        else:
            raise ValueError(
                "The multioutput parameter should be one of the following: " + "'raw_values', 'uniform_average'"
            )

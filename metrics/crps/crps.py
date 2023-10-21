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
    predictions: array-like of shape (n_samples, n_data) or (n_samples, n_data, n_timesteps, n_outputs)
        n_sampels from estimated target distribution.
    references: array-like of shape (n_data,) or (n_data, n_timesteps, n_outputs)
        Empirical (correct) target values from ground truth distribution.
    sum: bool, default=False
        Defines whether to sum over sum_axis dimension.
    sum_axis: int, default=-1
        Defines axis to sum over in case of multioutput input.
    multioutput: {"raw_values", "uniform_average"}
        Defines aggregating across the n_outputs dimension.
        "raw_values" returns full set of scores in case of multioutput input.
        "uniform_average" returns the average score across all outputs.
Returns:
    crps: float
        Continuous Ranked Probability Score.

Examples:    
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Crps(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(self._get_feature_types()),
            reference_urls=[
                "https://www.lokad.com/continuous-ranked-probability-score/"
            ],
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
        sum=False,
        sum_axis=-1,
        multioutput="uniform_average",
    ):
        pass

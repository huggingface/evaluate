# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

"""R squared metric."""


import datasets
import numpy as np

import evaluate


_CITATION = """
@article{williams2006relationship,
title={The relationship between R2 and the correlation coefficient},
author={Williams, James},
journal={Journal of Statistics Education},
volume={14},
number={2},
year={2006}
}
"""

_DESCRIPTION = """
R^2 (R Squared) is a statistical measure of the goodness of fit of a regression model. It represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

The R^2 value ranges from 0 to 1, with a higher value indicating a better fit. A value of 0 means that the model does not explain any of the variance in the dependent variable, while a value of 1 means that the model explains all of the variance.

R^2 can be calculated using the following formula:

r_squared = 1 - (Sum of Squared Errors / Sum of Squared Total)

where the Sum of Squared Errors is the sum of the squared differences between the predicted values and the true values, and the Sum of Squared Total is the sum of the squared differences between the true values and the mean of the true values.
"""

_KWARGS_DESCRIPTION = """
Computes the R Squared metric.

Args:
    predictions: List of predicted values of the dependent variable
    references: List of true values of the dependent variable
    zero_division: Which value to substitute as a metric value when encountering zero division. Should be one of 0, 1,
        "warn". "warn" acts as 0, but the warning is raised.

Returns:
    R^2 value ranging from 0 to 1, with a higher value indicating a better fit.

Examples:
    >>> r2_metric = evaluate.load("r_squared")
    >>> r2 = r2_metric.compute(predictions=[1, 2, 3, 4], references=[0.9, 2.1, 3.2, 3.8])
    >>> print(r2)
    0.95
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class R2(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float", id="sequence"),
                    "references": datasets.Value("float", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/scikit-learn/scikit-learn/"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/Coefficient_of_determination",
            ],
        )
        
    def _compute(self, predictions=None, references=None):
        """
        Computes the coefficient of determination (R-squared) of predictions with respect to references.

        Parameters:
            predictions (List or np.ndarray): The predicted values.
            references (List or np.ndarray): The true/reference values.

        Returns:
            float: The R-squared value.
        """
        predictions = np.array(predictions)
        references = np.array(references)
        
        # Calculate mean of the references
        mean_references = np.mean(references)

        # Calculate sum of squared residuals
        ssr = np.sum((predictions - references) ** 2)

        # Calculate sum of squared total
        sst = np.sum((references - mean_references) ** 2)

        # Calculate R Squared
        r_squared = 1 - (ssr / sst)

        return r_squared

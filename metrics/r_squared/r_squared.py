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


import warnings

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
    >>> r_squared = r2_metric.compute(predictions=[1, 2, 3, 4], references=[0.9, 2.1, 3.2, 3.8])
    >>> print(r_squared)
    0.98

    R^2 is undefined when the references have zero variance, since there is no
    variance to explain. `zero_division` selects what to return instead:

    >>> r2_metric.compute(predictions=[1, 2, 3, 4], references=[5, 5, 5, 5], zero_division=0)
    0.0
    >>> r2_metric.compute(predictions=[5, 5, 5, 5], references=[5, 5, 5, 5], zero_division=1)
    1.0
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class r_squared(evaluate.Metric):
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

    def _compute(self, predictions=None, references=None, zero_division="warn"):
        """
        Computes the coefficient of determination (R-squared) of predictions with respect to references.

        Parameters:
            predictions (List or np.ndarray): The predicted values.
            references (List or np.ndarray): The true/reference values.
            zero_division (0, 1 or "warn"): Value to return when `references` has zero
                variance, which makes the sum of squared total zero and R^2 undefined.
                "warn" returns 0.0 and raises a warning.

        Returns:
            float: The R-squared value, rounded to 3 decimal places.
        """
        if zero_division not in (0, 1, "warn"):
            raise ValueError(
                f'zero_division must be one of 0, 1 or "warn", got {zero_division!r}'
            )

        predictions = np.array(predictions)
        references = np.array(references)

        # Calculate mean of the references
        mean_references = np.mean(references)

        # Calculate sum of squared residuals
        ssr = np.sum((predictions - references) ** 2)

        # Calculate sum of squared total
        sst = np.sum((references - mean_references) ** 2)

        # R^2 is undefined when the references never vary: there is no variance to
        # explain, so the ratio divides by zero. Left to numpy this returns -inf
        # (or nan when the predictions match exactly), which then propagates
        # silently through any downstream aggregation.
        if sst == 0:
            if zero_division == "warn":
                warnings.warn(
                    "R^2 is undefined when all `references` are identical (zero "
                    "variance); returning 0.0. Pass zero_division=0 or 1 to choose "
                    "the value and silence this warning.",
                    UserWarning,
                    stacklevel=2,
                )
                return 0.0
            return float(zero_division)

        # Calculate R Squared
        r_squared = 1 - (ssr / sst)

        # Round off to 3 decimal places
        rounded_r_squared = round(r_squared, 3)

        return rounded_r_squared

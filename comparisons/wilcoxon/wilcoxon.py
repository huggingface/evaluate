# Copyright 2022 The HuggingFace Evaluate Authors
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
"""Wilcoxon test for model comparison."""

import datasets
from scipy.stats import wilcoxon

import evaluate


_DESCRIPTION = """
Wilcoxon's test is a non-parametric signed-rank test that tests whether the distribution of the differences is symmetric about zero. It can be used to compare the predictions of two models.
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions1 (`list` of `float`): Predictions for model 1.
    predictions2 (`list` of `float`): Predictions for model 2.

Returns:
    stat (`float`): Wilcoxon test score.
    p (`float`): The p value. Minimum possible value is 0. Maximum possible value is 1.0. A lower p value means a more significant difference.

Examples:
    >>> wilcoxon = evaluate.load("wilcoxon")
    >>> results = wilcoxon.compute(predictions1=[-7, 123.45, 43, 4.91, 5], predictions2=[1337.12, -9.74, 1, 2, 3.21])
    >>> print(results)
    {'stat': 5.0, 'p': 0.625}
"""


_CITATION = """
@incollection{wilcoxon1992individual,
  title={Individual comparisons by ranking methods},
  author={Wilcoxon, Frank},
  booktitle={Breakthroughs in statistics},
  pages={196--202},
  year={1992},
  publisher={Springer}
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Wilcoxon(evaluate.Comparison):
    def _info(self):
        return evaluate.ComparisonInfo(
            module_type="comparison",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions1": datasets.Value("float"),
                    "predictions2": datasets.Value("float"),
                }
            ),
        )

    def _compute(self, predictions1, predictions2):
        # calculate difference
        d = [p1 - p2 for (p1, p2) in zip(predictions1, predictions2)]

        # compute statistic
        res = wilcoxon(d)
        return {"stat": res.statistic, "p": res.pvalue}

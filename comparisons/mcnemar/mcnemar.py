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
"""McNemar test for model comparison."""

import datasets
from scipy.stats import chi2

import evaluate


_DESCRIPTION = """
McNemar's test is a diagnostic test over a contingency table resulting from the predictions of two classifiers. The test compares the sensitivity and specificity of the diagnostic tests on the same group reference labels. It can be computed with:
McNemar = (SE - SP)**2 / SE + SP
 Where:
SE: Sensitivity (Test 1 positive; Test 2 negative)
SP: Specificity (Test 1 negative; Test 2 positive)
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions1 (`list` of `int`): Predicted labels for model 1.
    predictions2 (`list` of `int`): Predicted labels for model 2.
    references (`list` of `int`): Ground truth labels.

Returns:
    p (`float` or `int`): McNemar test score. Minimum possible value is 0. Maximum possible value is 1.0. A lower p value means a more significant difference.

Examples:
    >>> mcnemar = evaluate.load("mcnemar")
    >>> results = mcnemar.compute(references=[1, 0, 1], predictions1=[1, 1, 1], predictions2=[1, 0, 1])
    >>> print(results)
    {'stat': 1.0, 'p': 0.31731050786291115}
"""


_CITATION = """
@article{mcnemar1947note,
  title={Note on the sampling error of the difference between correlated proportions or percentages},
  author={McNemar, Quinn},
  journal={Psychometrika},
  volume={12},
  number={2},
  pages={153--157},
  year={1947},
  publisher={Springer-Verlag}
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class McNemar(evaluate.EvaluationModule):
    def _info(self):
        return evaluate.EvaluationModuleInfo(
            module_type="comparison",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions1": datasets.Value("int64"),
                    "predictions2": datasets.Value("int64"),
                    "references": datasets.Value("int64"),
                }
            ),
        )

    def _compute(self, predictions1, predictions2, references):
        # construct contingency table
        tbl = [[0, 0], [0, 0]]
        for gt, p1, p2 in zip(references, predictions1, predictions2):
            if p1 == gt and p2 == gt:
                tbl[0][0] += 1
            elif p1 == gt:
                tbl[0][1] += 1
            elif p2 == gt:
                tbl[1][0] += 1
            else:
                tbl[1][1] += 1

        # compute statistic
        b, c = tbl[0][1], tbl[1][0]
        statistic = abs(b - c) ** 2 / (1.0 * (b + c))
        df = 1
        pvalue = chi2.sf(statistic, df)
        return {"stat": statistic, "p": pvalue}

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
"""Exact match test for model comparison."""

import datasets
import numpy as np

import evaluate


_DESCRIPTION = """
Returns the rate at which the predictions of one model exactly match those of another model.
"""


_KWARGS_DESCRIPTION = """
Args:
    predictions1 (`list` of `int`): Predicted labels for model 1.
    predictions2 (`list` of `int`): Predicted labels for model 2.

Returns:
    exact_match (`float`): Dictionary containing exact_match rate. Possible values are between 0.0 and 1.0, inclusive.

Examples:
    >>> exact_match = evaluate.load("exact_match", module_type="comparison")
    >>> results = exact_match.compute(predictions1=[1, 1, 1], predictions2=[1, 1, 1])
    >>> print(results)
    {'exact_match': 1.0}
"""


_CITATION = """
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ExactMatch(evaluate.EvaluationModule):
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
                }
            ),
        )

    def _compute(self, predictions1, predictions2):
        score_list = predictions1 == predictions2
        return {"exact_match": np.mean(score_list)}

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
"""An implementation for calculating CharCut, a character-based machine translation evaluation metric."""
from typing import Iterable, Union

import datasets
from charcut import calculate_charcut
from datasets import Sequence, Value

import evaluate


_CITATION = """\
@inproceedings{lardilleux-lepage-2017-charcut,
    title = "{CHARCUT}: Human-Targeted Character-Based {MT} Evaluation with Loose Differences",
    author = "Lardilleux, Adrien  and
      Lepage, Yves",
    booktitle = "Proceedings of the 14th International Conference on Spoken Language Translation",
    month = dec # " 14-15",
    year = "2017",
    address = "Tokyo, Japan",
    publisher = "International Workshop on Spoken Language Translation",
    url = "https://aclanthology.org/2017.iwslt-1.20",
    pages = "146--153"
}
"""

_DESCRIPTION = """\
CharCut compares outputs of MT systems with reference translations. The matching algorithm is based on an iterative
search for longest common substrings, combined with a length-based threshold that limits short and noisy character
matches. As a similarity metric this is not new, but to the best of our knowledge it was never applied to highlighting
and scoring of MT outputs. It has the neat effect of keeping character-based differences readable by humans."""

_KWARGS_DESCRIPTION = """
Calculates how good predictions are given some references. Predictions/references can be one or more sentences,
but they must be of the both type (one reference per hypothesis).
Args:
    predictions: a list of predictions to score. Each prediction should be a string with
     tokens separated by spaces.
    references: a list of reference for each prediction. Each reference should be a string with
     tokens separated by spaces.
Returns:
    charcut_mt: the CharCut score
Examples:
    >>> charcut_mt = evaluate.load("charcut_mt")
    >>> preds = ["this week the saudis denied information published in the new york times",
    ...          "this is in fact an estimate"]
    >>> refs = ["saudi arabia denied this week information published in the american new york times",
    ...         "this is actually an estimate"]
    >>> charcut_mt.compute(references=refs, predictions=preds)
    {'charcut_mt': 0.1971153846153846}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Charcut(evaluate.Metric):
    """Character-based MT evaluation."""

    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=[
                datasets.Features(
                    {"predictions": Value("string", id="prediction"), "references": Value("string", id="reference")}
                ),
            ],
            # Homepage of the module for documentation
            homepage="https://github.com/BramVanroy/CharCut",
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/BramVanroy/CharCut", "https://github.com/alardill/CharCut"],
        )

    def _compute(self, predictions: Iterable[str], references: Iterable[str]):
        return {"charcut_mt": calculate_charcut(predictions, references)[0]}

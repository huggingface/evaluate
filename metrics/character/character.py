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
"""An implementation for calculating CharacTER, a character-based TER variant, useful for machine translation tasks."""
import cer
import datasets
from datasets import Sequence, Value

import evaluate


_CITATION = """\
@inproceedings{wang-etal-2016-character,
    title = "{C}harac{T}er: Translation Edit Rate on Character Level",
    author = "Wang, Weiyue  and
      Peter, Jan-Thorsten  and
      Rosendahl, Hendrik  and
      Ney, Hermann",
    booktitle = "Proceedings of the First Conference on Machine Translation: Volume 2, Shared Task Papers",
    month = aug,
    year = "2016",
    address = "Berlin, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W16-2342",
    doi = "10.18653/v1/W16-2342",
    pages = "505--510",
}
"""

_DESCRIPTION = """\
CharacTer is a novel character level metric inspired by the commonly applied translation edit rate (TER). It is
defined as the minimum number of character edits required to adjust a hypothesis, until it completely matches the
reference, normalized by the length of the hypothesis sentence. CharacTer calculates the character level edit
distance while performing the shift edit on word level. Unlike the strict matching criterion in TER, a hypothesis
word is considered to match a reference word and could be shifted, if the edit distance between them is below a
threshold value. The Levenshtein distance between the reference and the shifted hypothesis sequence is computed on the
character level. In addition, the lengths of hypothesis sequences instead of reference sequences are used for
normalizing the edit distance, which effectively counters the issue that shorter translations normally achieve lower
TER."""

_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: a single prediction or a list of predictions to score. Each prediction should be a string with
     tokens separated by spaces.
    references: a single reference or a list of references for each prediction. Each reference should be a string with
     tokens separated by spaces.
Returns:
    cer_scores: a list of all scores, one per ref/hyp pair
Examples:
    >>> character_mt = evaluate.load("character")
    >>> preds = ["this week the saudis denied information published in the new york times"]
    >>> refs = ["saudi arabia denied this week information published in the american new york times"]
    >>> character_mt.compute(references=refs, predictions=preds)
    {
        'cer_scores': [0.36619718309859156]
    }
    >>> preds = ["this week the saudis denied information published in the new york times",
    ...          "this is in fact an estimate"]
    >>> refs = ["saudi arabia denied this week information published in the american new york times",
    ...         "this is actually an estimate"]
    >>> character_mt.compute(references=refs, predictions=preds)
    {
        'cer_scores': [0.36619718309859156, 0.25925925925925924]
    }
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Character(evaluate.Metric):
    """CharacTer is a novel character level metric inspired by the commonly applied translation edit rate (TER)."""

    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {"predictions": Value("string", id="prediction"), "references": Value("string", id="reference")}
                ),
                datasets.Features(
                    {
                        "predictions": Sequence(Value("string", id="prediction"), id="predictions"),
                        "references": Sequence(Value("string", id="reference"), id="references"),
                    }
                ),
            ],
            homepage="https://github.com/bramvanroy/CharacTER",
            codebase_urls=["https://github.com/bramvanroy/CharacTER", "https://github.com/rwth-i6/CharacTER"],
        )

    def _compute(self, predictions, references):
        """Returns the scores. When more than one prediction/reference is given, we can use
        the corpus-focused metric"""
        if isinstance(predictions, str):
            predictions = [predictions]

        if isinstance(references, str):
            references = [references]

        predictions = [p.split() for p in predictions]
        references = [r.split() for r in references]

        return {"cer_scores": cer.calculate_cer_corpus(predictions, references)["cer_scores"]}

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
from typing import Iterable, Literal

import cer
import datasets
from datasets import Value

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
Calculates how good the predictions are in terms of the CharacTER metric given some references
Args:
    predictions: a list of predictions to score. Each prediction should be a string with
     tokens separated by spaces.
    references: a list of references for each prediction. Each reference should be a string with
     tokens separated by spaces.
    aggregate: one of "mean", "sum", "median" to indicate how the scores of individual sentences should be
     aggregated
    return_all_scores: a boolean, indicating whether in addition to the aggregated score, also all individual
     scores should be returned
Returns:
    cer_score: an aggregated score across all the items, based on 'aggregate'
    cer_scores: (optionally, if 'return_all_scores' evaluates to True) a list of all scores, one per ref/hyp pair
Examples:
    >>> character_mt = evaluate.load("character")
    >>> preds = ["this week the saudis denied information published in the new york times"]
    >>> refs = ["saudi arabia denied this week information published in the american new york times"]
    >>> character_mt.compute(references=refs, predictions=preds)
    {'cer_score': 0.36619718309859156}
    >>> preds = ["this week the saudis denied information published in the new york times",
    ...          "this is in fact an estimate"]
    >>> refs = ["saudi arabia denied this week information published in the american new york times",
    ...         "this is actually an estimate"]
    >>> character_mt.compute(references=refs, predictions=preds, aggregate="sum", return_all_scores=True)
    {'cer_score': 0.6254564423578508, 'cer_scores': [0.36619718309859156, 0.25925925925925924]}
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
            ],
            homepage="https://github.com/bramvanroy/CharacTER",
            codebase_urls=["https://github.com/bramvanroy/CharacTER", "https://github.com/rwth-i6/CharacTER"],
        )

    def _compute(
        self,
        predictions: Iterable[str],
        references: Iterable[str],
        aggregate: Literal["mean", "sum", "median"] = "mean",
        return_all_scores: bool = False,
    ):
        """Returns the scores. When more than one prediction/reference is given, we can use
        the corpus-focused metric"""
        predictions = [p.split() for p in predictions]
        references = [r.split() for r in references]

        scores_d = cer.calculate_cer_corpus(predictions, references)
        cer_scores = scores_d["cer_scores"]

        if aggregate == "sum":
            score = sum(cer_scores)
        elif aggregate == "mean":
            score = scores_d["mean"]
        elif aggregate == "median":
            score = scores_d["median"]
        else:
            raise ValueError("'aggregate' must be one of 'sum', 'mean', 'median'")

        if return_all_scores:
            return {"cer_score": score, "cer_scores": cer_scores}
        else:
            return {"cer_score": score}

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
"""CharacTER metric, a character-based TER variant, for machine translation."""
import math
from statistics import mean, median
from typing import Iterable, List, Union

import cer
import datasets
from cer import calculate_cer
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
CharacTer is a character-level metric inspired by the commonly applied translation edit rate (TER). It is
defined as the minimum number of character edits required to adjust a hypothesis, until it completely matches the
reference, normalized by the length of the hypothesis sentence. CharacTer calculates the character level edit
distance while performing the shift edit on word level. Unlike the strict matching criterion in TER, a hypothesis
word is considered to match a reference word and could be shifted, if the edit distance between them is below a
threshold value. The Levenshtein distance between the reference and the shifted hypothesis sequence is computed on the
character level. In addition, the lengths of hypothesis sequences instead of reference sequences are used for
normalizing the edit distance, which effectively counters the issue that shorter translations normally achieve lower
TER."""

_KWARGS_DESCRIPTION = """
Calculates how good the predictions are in terms of the CharacTER metric given some references.
Args:
    predictions: a list of predictions to score. Each prediction should be a string with
     tokens separated by spaces.
    references: a list of references for each prediction. You can also pass multiple references for each prediction,
     so a list and in that list a sublist for each prediction for its related references. When multiple references are
     given, the lowest (best) score is returned for that prediction-references pair.
     Each reference should be a string with tokens separated by spaces.
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
    >>> preds = ["this week the saudis denied information published in the new york times"]
    >>> refs = [["saudi arabia denied this week information published in the american new york times",
    ...          "the saudis have denied new information published in the ny times"]]
    >>> character_mt.compute(references=refs, predictions=preds, aggregate="median", return_all_scores=True)
    {'cer_score': 0.36619718309859156, 'cer_scores': [0.36619718309859156]}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Character(evaluate.Metric):
    """CharacTer is a character-level metric inspired by the commonly applied translation edit rate (TER)."""

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
                        "predictions": Value("string", id="prediction"),
                        "references": Sequence(Value("string", id="reference"), id="references"),
                    }
                ),
            ],
            homepage="https://github.com/bramvanroy/CharacTER",
            codebase_urls=["https://github.com/bramvanroy/CharacTER", "https://github.com/rwth-i6/CharacTER"],
        )

    def _compute(
        self,
        predictions: Iterable[str],
        references: Union[Iterable[str], Iterable[Iterable[str]]],
        aggregate: str = "mean",
        return_all_scores: bool = False,
    ):
        if aggregate not in ("mean", "sum", "median"):
            raise ValueError("'aggregate' must be one of 'sum', 'mean', 'median'")

        predictions = [p.split() for p in predictions]
        # Predictions and references have the same internal types (both lists of strings),
        # so only one reference per prediction
        if isinstance(references[0], str):
            references = [r.split() for r in references]

            scores_d = cer.calculate_cer_corpus(predictions, references)
            cer_scores: List[float] = scores_d["cer_scores"]

            if aggregate == "sum":
                score = sum(cer_scores)
            elif aggregate == "mean":
                score = scores_d["mean"]
            else:
                score = scores_d["median"]
        else:
            # In the case of multiple references, we just find the "best score",
            # i.e., the reference that the prediction is closest to, i.e. the lowest characTER score
            references = [[r.split() for r in refs] for refs in references]

            cer_scores = []
            for pred, refs in zip(predictions, references):
                min_score = math.inf
                for ref in refs:
                    score = calculate_cer(pred, ref)

                    if score < min_score:
                        min_score = score

                cer_scores.append(min_score)

            if aggregate == "sum":
                score = sum(cer_scores)
            elif aggregate == "mean":
                score = mean(cer_scores)
            else:
                score = median(cer_scores)

        # Return scores
        if return_all_scores:
            return {"cer_score": score, "cer_scores": cer_scores}
        else:
            return {"cer_score": score}

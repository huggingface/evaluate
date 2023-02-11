# Copyright 2020 The HuggingFace Evaluate Authors.
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
"""ANLS - Average Normalized Levenshtein Similarity"""

import datasets
import evaluate

from compute_score import compute_score


_CITATION = """\
@article{,
    title = {Binary codes capable of correcting deletions, insertions, and reversals},
    journal = {Soviet physics doklady},
    volume = {10},
    number = {8},
    pages = {707--710},
    year = {1966},
    url = {https://nymity.ch/sybilhunting/pdf/Levenshtein1966a.pdf},
    author = {V. I. Levenshtein},
}
"""

_DESCRIPTION = """\
ANLS refer to the average normalized Levenshtein similarity.
"""

_KWARGS_DESCRIPTION = """
Computes Average Normalized Levenshtein Similarity (ANLS).
Args:
    predictions: List of question-answers dictionaries with the following key-values:
    - 'question_id': id of the question-answer pair as given in the references
    (see below)
    - 'prediction_text': the text of the answer
    references: List of question-answers dictionaries with the following key-values:
    - 'question_id': id of the question-answer pair (see above)
    - 'answers': list of possible texts for the answer, as a list of strings
Returns:
    'anls': The ANLS score of predicted tokens versus the gold answer
Examples:
    >>> predictions = [{'prediction_text': 'Denver Broncos',
                        'question_id': '56e10a3be'}]
    >>> references = [{'answers': ['Denver Broncos', 'Denver R. Broncos'],
                       'question_id': '56e10a3be'}]
    >>> anls_metric = evaluate.load("anls")
    >>> results = anls_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'anls_score': 100.0}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Anls(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {"question_id": datasets.Value("string"),
                                    "prediction_text": datasets.Value("string")},
                    "references": {
                        "question_id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(datasets.Value("string")),
                    },
                }
            )
        )

    def _compute(self, predictions, references):
        ground_truths = {x['question_id']: x['answers'] for x in references}
        predictions = {x['question_id']: x['prediction_text'] for x in predictions}
        anls_score = compute_score(predictions=predictions, ground_truths=ground_truths)
        return {"anls_score": anls_score}

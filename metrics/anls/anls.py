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
""" ROUGE metric."""

import datasets

import evaluate



_CITATION = """\
@INPROCEEDINGS{ANLS,
    author = {Vladimir Levenshtein},
    title = {Binary Codes Capable of Correcting Deletions, Insertions, and Reversals},
    booktitle = {},
    year = {1965},
    pages = {}
}
"""


_DESCRIPTION = """\
The average normalized Levenshtein similarity (ANLS) is a string similarity metric 
that measures the difference between two strings. It is based on the Levenshtein 
distance, which is a measure of the minimum number of edit operations (insertions, 
deletions, and substitutions) needed to transform one string into another.
"""

_KWARGS_DESCRIPTION = """
 Args:
      targets: list of lists of texts containing the String targets.
      prediction: list of Text containing the predictions.
    Returns:
      A float for the score.
Examples:

>>> targets = [["hello my friend", "friend", "hi"], ["leave"], ["bad","okay"]]
>>> prediction = ["hi friend", "I am leaving", "that's good"]
>>> score = average_normalized_Levenshtein_similarity(targets, prediction)
>>> print(score)
0.73333                      
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ANLS(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            
            
        )
    
    def _compute(self, references, predictions):

        def levenshtein_similarity(s1, s2):
            m = len(s1)
            n = len(s2)
            d = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        d[i][j] = d[i - 1][j - 1]
                    else:
                        d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
            l = max(m, n)
            return 1 - (float(d[m][n]) / l)




        N = len(predictions)
        score = 0
        for an in range(len(predictions)):
            best_answer = []
            for gt in range(len(references[an])):
                answer = levenshtein_similarity(references[an][gt], predictions[gt])
                best_answer.append(answer)
                score += max(best_answer)

        return round(score/N,5)
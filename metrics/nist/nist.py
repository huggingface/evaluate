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
"""NLTK's NIST implementation on both the sentence and corpus level"""
import datasets
from datasets import Sequence, Value
from nltk.translate.nist_score import corpus_nist, sentence_nist

import evaluate


_CITATION = """\
@inproceedings{10.5555/1289189.1289273,
    author = {Doddington, George},
    title = {Automatic Evaluation of Machine Translation Quality Using N-Gram Co-Occurrence Statistics},
    year = {2002},
    publisher = {Morgan Kaufmann Publishers Inc.},
    address = {San Francisco, CA, USA},
    booktitle = {Proceedings of the Second International Conference on Human Language Technology Research},
    pages = {138–145},
    numpages = {8},
    location = {San Diego, California},
    series = {HLT '02}
}
"""

_DESCRIPTION = """\
    DARPA commissioned NIST to develop an MT evaluation facility based on the BLEU
    score. The official script used by NIST to compute BLEU and NIST score is
    mteval-14.pl. The main differences are:

     - BLEU uses geometric mean of the ngram overlaps, NIST uses arithmetic mean.
     - NIST has a different brevity penalty
     - NIST score from mteval-14.pl has a self-contained tokenizer
"""


_KWARGS_DESCRIPTION = """
Computes NIST score of translated segments against one or more references.
Args:
    predictions: tokenized predictions to score. For sentence-level NIST, a list of tokens (str);
     for corpus-level NIST, a list (sentences) of lists of tokens (str)
    references:  potentially multiple tokenized references for each prediction.  For sentence-level NIST, a
     list (multiple potential references) of list of tokens (str); for corpus-level NIST, a list (corpus) of lists
     (multiple potential references) of lists of tokens (str)
    n: highest n-gram order
Returns:
    'nist': nist score 
Examples:
    >>> nist = evaluate.load("nist")
    >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
    ...               'ensures', 'that', 'the', 'military', 'always',
    ...               'obeys', 'the', 'commands', 'of', 'the', 'party']

    >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
    ...               'ensures', 'that', 'the', 'military', 'will', 'forever',
    ...               'heed', 'Party', 'commands']

    >>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
    ...               'guarantees', 'the', 'military', 'forces', 'always',
    ...               'being', 'under', 'the', 'command', 'of', 'the',
    ...               'Party']

    >>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
    ...               'army', 'always', 'to', 'heed', 'the', 'directions',
    ...               'of', 'the', 'party']
    
    >>> nist.compute(hypothesis1, [reference1, reference2])
    {'nist': 3.3709935957649324}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NIST(evaluate.Metric):
    """A wrapper around NLTK's NIST implementation."""

    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": Sequence(Value("string", id="token"), id="prediction"),
                        "references": Sequence(Sequence(Value("string", id="token"), id="reference"), id="references"),
                    }
                ),
                datasets.Features(
                    {
                        "predictions": Sequence(
                            Sequence(Value("string", id="token"), id="prediction"), id="predictions"
                        ),
                        "references": Sequence(
                            Sequence(Sequence(Value("string", id="token"), id="reference"), id="references"),
                            id="reference_corpus",
                        ),
                    }
                ),
            ],
            homepage="https://www.nltk.org/api/nltk.translate.nist_score.html",
            codebase_urls=["https://github.com/nltk/nltk/blob/develop/nltk/translate/nist_score.py"],
            reference_urls=["https://en.wikipedia.org/wiki/NIST_(metric)"],
        )

    def _compute(self, predictions, references, n: int = 5):
        try:
            if isinstance(predictions[0], str) and isinstance(references[0][0], str):
                return {"nist": sentence_nist(references=references, hypothesis=predictions, n=n)}
            elif isinstance(predictions[0][0], str) and isinstance(references[0][0][0], str):
                return {"nist": corpus_nist(list_of_references=references, hypotheses=predictions, n=n)}
            else:
                raise TypeError
        except (TypeError, AssertionError) as exc:
            raise TypeError(
                "Invalid input format given. Note that order of arguments is different from the original"
                " NLTK signature. Here, the predictions come first and references second.\nNIST score"
                " (as per NLTK's implementation) allows two formats, sentence level and corpus level. For"
                " sentence: predictions (List[str]) and potentially multiple references (List[List[str]])."
                " For corpus: predictions (List[List[str]]) and references (List[List[List[str]]])."
            ) from exc

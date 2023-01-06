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
from typing import Dict, Optional

import datasets
import nltk
from datasets import Sequence, Value


try:
    nltk.data.find("perluniprops")
except LookupError:
    nltk.download("perluniprops", quiet=True)  # NISTTokenizer requirement

from nltk.tokenize.nist import NISTTokenizer
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

 - BLEU uses geometric mean of the ngram precisions, NIST uses arithmetic mean.
 - NIST has a different brevity penalty
 - NIST score from mteval-14.pl has a self-contained tokenizer (in the Hugging Face implementation we rely on NLTK's
implementation of the NIST-specific tokenizer)
"""


_KWARGS_DESCRIPTION = """
Computes NIST score of translated segments against one or more references.
Args:
    predictions: predictions to score (list of str)
    references: potentially multiple references for each prediction (list of str or list of list of str)
    n: highest n-gram order
    lowercase: whether to lowercase the data (only applicable if 'western_lang' is True)
    western_lang: whether the current language is a Western language, which will enable some specific tokenization
 rules with respect to, e.g., punctuation

Returns:
    'nist_mt': nist_mt score
Examples:
    >>> nist_mt = evaluate.load("nist_mt")
    >>> hypothesis = "It is a guide to action which ensures that the military always obeys the commands of the party"
    >>> reference1 = "It is a guide to action that ensures that the military will forever heed Party commands"
    >>> reference2 = "It is the guiding principle which guarantees the military forces always being under the command of the Party"
    >>> reference3 = "It is the practical guide for the army always to heed the directions of the party"
    >>> nist_mt.compute(predictions=[hypothesis], references=[[reference1, reference2, reference3]])
    {'nist_mt': 3.3709935957649324}
    >>> nist_mt.compute(predictions=[hypothesis], references=[reference1])
    {'nist_mt': 2.4477124183006533}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NistMt(evaluate.Metric):
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
                        "predictions": Value("string", id="prediction"),
                        "references": Sequence(Value("string", id="reference"), id="references"),
                    }
                ),
                datasets.Features(
                    {"predictions": Value("string", id="prediction"), "references": Value("string", id="reference")}
                ),
            ],
            homepage="https://www.nltk.org/api/nltk.translate.nist_score.html",
            codebase_urls=["https://github.com/nltk/nltk/blob/develop/nltk/translate/nist_score.py"],
            reference_urls=["https://en.wikipedia.org/wiki/NIST_(metric)"],
        )

    def _compute(self, predictions, references, n: int = 5, lowercase=False, western_lang=True):
        tokenizer = NISTTokenizer()

        # Account for single reference cases: references always need to have one more dimension than predictions
        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        predictions = [
            tokenizer.tokenize(pred, return_str=False, lowercase=lowercase, western_lang=western_lang)
            for pred in predictions
        ]
        references = [
            [
                tokenizer.tokenize(ref, return_str=False, lowercase=lowercase, western_lang=western_lang)
                for ref in ref_sentences
            ]
            for ref_sentences in references
        ]
        return {"nist_mt": corpus_nist(list_of_references=references, hypotheses=predictions, n=n)}

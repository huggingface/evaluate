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
""" Google BLEU (aka GLEU) metric. """

from typing import Dict, List

import datasets
from nltk.translate import gleu_score

import evaluate
from evaluate import EvaluationModuleInfo

from .tokenizer_13a import Tokenizer13a


_CITATION = """\
@misc{wu2016googles,
      title={Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation},
      author={Yonghui Wu and Mike Schuster and Zhifeng Chen and Quoc V. Le and Mohammad Norouzi and Wolfgang Macherey
              and Maxim Krikun and Yuan Cao and Qin Gao and Klaus Macherey and Jeff Klingner and Apurva Shah and Melvin
              Johnson and Xiaobing Liu and Łukasz Kaiser and Stephan Gouws and Yoshikiyo Kato and Taku Kudo and Hideto
              Kazawa and Keith Stevens and George Kurian and Nishant Patil and Wei Wang and Cliff Young and
              Jason Smith and Jason Riesa and Alex Rudnick and Oriol Vinyals and Greg Corrado and Macduff Hughes
              and Jeffrey Dean},
      year={2016},
      eprint={1609.08144},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
The BLEU score has some undesirable properties when used for single
sentences, as it was designed to be a corpus measure. We therefore
use a slightly different score for our RL experiments which we call
the 'GLEU score'. For the GLEU score, we record all sub-sequences of
1, 2, 3 or 4 tokens in output and target sequence (n-grams). We then
compute a recall, which is the ratio of the number of matching n-grams
to the number of total n-grams in the target (ground truth) sequence,
and a precision, which is the ratio of the number of matching n-grams
to the number of total n-grams in the generated output sequence. Then
GLEU score is simply the minimum of recall and precision. This GLEU
score's range is always between 0 (no matches) and 1 (all match) and
it is symmetrical when switching output and target. According to
our experiments, GLEU score correlates quite well with the BLEU
metric on a corpus level but does not have its drawbacks for our per
sentence reward objective.
"""

_KWARGS_DESCRIPTION = """\
Computes corpus-level Google BLEU (GLEU) score of translated segments against one or more references.
Instead of averaging the sentence level GLEU scores (i.e. macro-average precision), Wu et al. (2016) sum up the matching
tokens and the max of hypothesis and reference tokens for each sentence, then compute using the aggregate values.

Args:
    predictions (list of str): list of translations to score.
    references (list of list of str): list of lists of references for each translation.
    tokenizer : approach used for tokenizing `predictions` and `references`.
        The default tokenizer is `tokenizer_13a`, a minimal tokenization approach that is equivalent to `mteval-v13a`, used by WMT.
        This can be replaced by any function that takes a string as input and returns a list of tokens as output.
    min_len (int): The minimum order of n-gram this function should extract. Defaults to 1.
    max_len (int): The maximum order of n-gram this function should extract. Defaults to 4.

Returns:
    'google_bleu': google_bleu score

Examples:
    Example 1:
        >>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', \
        'he read the book because he was interested in world history']
        >>> references = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat'], \
        ['he was interested in world history because he read the book']]
        >>> google_bleu = evaluate.load("google_bleu")
        >>> results = google_bleu.compute(predictions=predictions, references=references)
        >>> print(round(results["google_bleu"], 2))
        0.44

    Example 2:
        >>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', \
        'he read the book because he was interested in world history']
        >>> references = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat', \
        'It is a guide to action that ensures that the rubber duck will never heed the cat commands', \
        'It is the practical guide for the rubber duck army never to heed the directions of the cat'], \
        ['he was interested in world history because he read the book']]
        >>> google_bleu = evaluate.load("google_bleu")
        >>> results = google_bleu.compute(predictions=predictions, references=references)
        >>> print(round(results["google_bleu"], 2))
        0.61

    Example 3:
        >>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', \
        'he read the book because he was interested in world history']
        >>> references = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat', \
        'It is a guide to action that ensures that the rubber duck will never heed the cat commands', \
        'It is the practical guide for the rubber duck army never to heed the directions of the cat'], \
        ['he was interested in world history because he read the book']]
        >>> google_bleu = evaluate.load("google_bleu")
        >>> results = google_bleu.compute(predictions=predictions, references=references, min_len=2)
        >>> print(round(results["google_bleu"], 2))
        0.53

    Example 4:
        >>> predictions = ['It is a guide to action which ensures that the rubber duck always disobeys the commands of the cat', \
        'he read the book because he was interested in world history']
        >>> references = [['It is the guiding principle which guarantees the rubber duck forces never being under the command of the cat', \
        'It is a guide to action that ensures that the rubber duck will never heed the cat commands', \
        'It is the practical guide for the rubber duck army never to heed the directions of the cat'], \
        ['he was interested in world history because he read the book']]
        >>> google_bleu = evaluate.load("google_bleu")
        >>> results = google_bleu.compute(predictions=predictions,references=references, min_len=2, max_len=6)
        >>> print(round(results["google_bleu"], 2))
        0.4
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class GoogleBleu(evaluate.EvaluationModule):
    def _info(self) -> EvaluationModuleInfo:
        return evaluate.EvaluationModuleInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                }
            ),
        )

    def _compute(
        self,
        predictions: List[str],
        references: List[List[str]],
        tokenizer=Tokenizer13a(),
        min_len: int = 1,
        max_len: int = 4,
    ) -> Dict[str, float]:
        references = [[tokenizer(r) for r in ref] for ref in references]
        predictions = [tokenizer(p) for p in predictions]
        return {
            "google_bleu": gleu_score.corpus_gleu(
                list_of_references=references, hypotheses=predictions, min_len=min_len, max_len=max_len
            )
        }

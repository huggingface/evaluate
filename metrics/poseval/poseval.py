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
""" seqeval metric. """

from typing import Union

import datasets
from sklearn.metrics import classification_report

import evaluate


_CITATION = """\
@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}
"""

_DESCRIPTION = """\
The poseval metric can be used to evaluate POS taggers. Since seqeval does not work well with POS data \
(see e.g. [here](https://stackoverflow.com/questions/71327693/how-to-disable-seqeval-label-formatting-for-pos-tagging))\
that is not in IOB format the poseval metric is an alternative. It treats each token in the dataset as independant \
observation and computes the precision, recall and F1-score irrespective of sentences. It uses scikit-learns's \
[classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) \
to compute the scores.

"""

_KWARGS_DESCRIPTION = """
Computes the poseval metric.

Args:
    predictions: List of List of predicted labels (Estimated targets as returned by a tagger)
    references: List of List of reference labels (Ground truth (correct) target values)
    zero_division: Which value to substitute as a metric value when encountering zero division. Should be on of 0, 1,
        "warn". "warn" acts as 0, but the warning is raised.

Returns:
    'scores': dict. Summary of the scores for overall and per type
        Overall (weighted and macro avg):
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure,
        Per type:
            'precision': precision,
            'recall': recall,
            'f1': F1 score, also known as balanced F-score or F-measure
Examples:

    >>> predictions = [['INTJ', 'ADP', 'PROPN', 'NOUN', 'PUNCT', 'INTJ', 'ADP', 'PROPN', 'VERB', 'SYM']]
    >>> references = [['INTJ', 'ADP', 'PROPN', 'PROPN', 'PUNCT', 'INTJ', 'ADP', 'PROPN', 'PROPN', 'SYM']]
    >>> poseval = evaluate.load("poseval")
    >>> results = poseval.compute(predictions=predictions, references=references)
    >>> print(list(results.keys()))
    ['ADP', 'INTJ', 'NOUN', 'PROPN', 'PUNCT', 'SYM', 'VERB', 'accuracy', 'macro avg', 'weighted avg']
    >>> print(results["accuracy"])
    0.8
    >>> print(results["PROPN"]["recall"])
    0.5
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Poseval(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://scikit-learn.org",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/scikit-learn/scikit-learn"],
        )

    def _compute(
        self,
        predictions,
        references,
        zero_division: Union[str, int] = "warn",
    ):
        report = classification_report(
            y_true=[label for ref in references for label in ref],
            y_pred=[label for pred in predictions for label in pred],
            output_dict=True,
            zero_division=zero_division,
        )

        return report

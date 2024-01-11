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
"""Confusion Matrix."""

import datasets
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

import evaluate


_DESCRIPTION = """
The confusion matrix evaluates classification accuracy. Each row in a confusion matrix represents a true class and each column represents the instances in a predicted class
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`list` of `int`): Predicted labels.
    references (`list` of `int`): Ground truth labels.
    labels (`list` of `int`): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
    sample_weight (`list` of `float`): Sample weights.
    normalize (`str`): Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.

Returns:
    confusion_matrix (`list` of `list` of `int`): Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
    In a multilabel scenario, each element in the confusion matrix represents the number of samples that have been assigned a particular combination of labels.

Examples:

    Example 1-A simple example
        >>> confusion_matrix_metric = evaluate.load("confusion_matrix")
        >>> results = confusion_matrix_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
        >>> print(results)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'confusion_matrix': array([[1, 0, 1], [0, 2, 0], [1, 1, 0]][...])}

    Example 2-Multilabel scenario
        >>> # you must pass (config_name="multilabel") to the load method
        >>> confusion_matrix_metric = evaluate.load("confusion_matrix", config_name="multilabel")
        >>> results = confusion_matrix_metric.compute(references=[[0, 1], [1, 0], [0, 0], [0, 1], [1, 0], [0, 0]], predictions=[[0, 1], [1, 0], [1, 0], [0, 0], [1, 0], [0, 1]])
        >>> print(results)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'confusion_matrix': [[[3, 1], [0, 2]], [[3, 1], [1, 1]]]}
"""


_CITATION = """
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


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ConfusionMatrix(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html",
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html",
            ],
        )

    def _compute(self, predictions, references, labels=None, sample_weight=None, normalize=None):
        if self.config_name == "multilabel":
            return {
                "confusion_matrix": multilabel_confusion_matrix(
                    references,
                    predictions,
                    sample_weight=sample_weight,
                    labels=labels,
                    samplewise=normalize == "samplewise",
                ),
            }
        return {
            "confusion_matrix": confusion_matrix(
                references, predictions, labels=labels, sample_weight=sample_weight, normalize=normalize
            ),
        }

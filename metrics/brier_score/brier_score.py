# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Brier Score Metric"""

import datasets
from sklearn.metrics import brier_score_loss

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
Brier score is a type of evaluation metric for classification tasks, where you predict outcomes such as win/lose, spam/ham, click/no-click etc.
`BrierScore = 1/N * sum( (p_i - o_i)^2 )`
"""


_KWARGS_DESCRIPTION = """
Args:
    y_true : array of shape (n_samples,)
        True targets.

    y_prob : array of shape (n_samples,)
        Probabilities of the positive class.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    pos_label : int or str, default=None
        Label of the positive class. `pos_label` will be inferred in the
        following manner:

        * if `y_true` in {-1, 1} or {0, 1}, `pos_label` defaults to 1;
        * else if `y_true` contains string, an error will be raised and
          `pos_label` should be explicitly specified;
        * otherwise, `pos_label` defaults to the greater label,
          i.e. `np.unique(y_true)[-1]`.

Returns
    score : float
        Brier score loss.
Examples:
    Example-1: if y_true in {-1, 1} or {0, 1}, pos_label defaults to 1.
        >>> import numpy as np
        >>> brier_score = evaluate.load("brier_score")
        >>> references = np.array([0, 0, 1, 1])
        >>> predictions = np.array([0.1, 0.9, 0.8, 0.3])
        >>> results = brier_score.compute(references=references, predictions=predictions)
        >>> print(round(results["brier_score"], 4))
        0.3375

    Example-2: if y_true contains string, an error will be raised and pos_label should be explicitly specified.
        >>> import numpy as np
        >>> brier_score = evaluate.load("brier_score")
        >>> references =  np.array(["spam", "ham", "ham", "spam"])
        >>> predictions = np.array([0.1, 0.9, 0.8, 0.3])
        >>> results = brier_score.compute(references=references, predictions=predictions, pos_label="ham")
        >>> print(round(results["brier_score"], 4))
        0.0375
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BrierScore(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._get_feature_types(),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html"],
        )

    def _get_feature_types(self):
        if self.config_name == "multilist":
            return [
                datasets.Features(
                    {
                        "references": datasets.Sequence(datasets.Value("float")),
                        "predictions": datasets.Sequence(datasets.Value("float")),
                    }
                ),
                datasets.Features(
                    {
                        "references": datasets.Sequence(datasets.Value("string")),
                        "predictions": datasets.Sequence(datasets.Value("float")),
                    }
                ),
            ]
        else:
            return [
                datasets.Features(
                    {
                        "references": datasets.Value("float"),
                        "predictions": datasets.Value("float"),
                    }
                ),
                datasets.Features(
                    {
                        "references": datasets.Value("string"),
                        "predictions": datasets.Value("float"),
                    }
                ),
            ]

    def _compute(self, references, predictions, sample_weight=None, pos_label=1):

        brier_score = brier_score_loss(references, predictions, sample_weight=sample_weight, pos_label=pos_label)

        return {"brier_score": brier_score}

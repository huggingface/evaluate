# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Matthews Correlation metric."""

import datasets
import numpy as np
from sklearn.metrics import matthews_corrcoef

import evaluate


_DESCRIPTION = """
Compute the Matthews correlation coefficient (MCC)

The Matthews correlation coefficient is used in machine learning as a
measure of the quality of binary and multiclass classifications. It takes
into account true and false positives and negatives and is generally
regarded as a balanced measure which can be used even if the classes are of
very different sizes. The MCC is in essence a correlation coefficient value
between -1 and +1. A coefficient of +1 represents a perfect prediction, 0
an average random prediction and -1 an inverse prediction.  The statistic
is also known as the phi coefficient. [source: Wikipedia]
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (list of int): Predicted labels, as returned by a model.
    references (list of int): Ground truth labels.
    average (`string`): This parameter is used for multilabel configs. Defaults to `None`.
        - None (default): Returns an array of Matthews correlation coefficients, one for each feature
        - 'macro': Calculate metrics for each feature, and find their unweighted mean.
    sample_weight (list of int, float, or bool): Sample weights. Defaults to `None`.
Returns:
    matthews_correlation (dict containing float): Matthews correlation.
Examples:
    Example 1, a basic example with only predictions and references as inputs:
        >>> matthews_metric = evaluate.load("matthews_correlation")
        >>> results = matthews_metric.compute(references=[1, 3, 2, 0, 3, 2],
        ...                                     predictions=[1, 2, 2, 0, 3, 3])
        >>> print(round(results['matthews_correlation'], 2))
        0.54

    Example 2, the same example as above, but also including sample weights:
        >>> matthews_metric = evaluate.load("matthews_correlation")
        >>> results = matthews_metric.compute(references=[1, 3, 2, 0, 3, 2],
        ...                                     predictions=[1, 2, 2, 0, 3, 3],
        ...                                     sample_weight=[0.5, 3, 1, 1, 1, 2])
        >>> print(round(results['matthews_correlation'], 2))
        0.1

    Example 3, the same example as above, but with sample weights that cause a negative correlation:
        >>> matthews_metric = evaluate.load("matthews_correlation")
        >>> results = matthews_metric.compute(references=[1, 3, 2, 0, 3, 2],
        ...                                     predictions=[1, 2, 2, 0, 3, 3],
        ...                                     sample_weight=[0.5, 1, 0, 0, 0, 1])
        >>> print(round(results['matthews_correlation'], 2))
        -0.25
    Example 4, Multi-label without averaging:
        >>> matthews_metric = evaluate.load("matthews_correlation", config_name="multilabel")
        >>> results = matthews_metric.compute(references=[[0,1], [1,0], [1,1]],
        ...                                     predictions=[[0,1], [1,1], [0,1]])
        >>> print(results['matthews_correlation'])
        [0.5, 0.0]
    Example 5, Multi-label with averaging:
        >>> matthews_metric = evaluate.load("matthews_correlation", config_name="multilabel")
        >>> results = matthews_metric.compute(references=[[0,1], [1,0], [1,1]],
        ...                                     predictions=[[0,1], [1,1], [0,1]],
        ...                                     average='macro')
        >>> print(round(results['matthews_correlation'], 2))
        0.25
"""

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


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MatthewsCorrelation(evaluate.Metric):
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
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html"
            ],
        )

    def _compute(self, predictions, references, average=None, sample_weight=None):
        if self.config_name == "multilabel":
            references = np.array(references)
            predictions = np.array(predictions)
            if references.ndim != 2 or predictions.ndim != 2:
                raise ValueError("For multi-label inputs, both references and predictions should be 2-dimensional")
            matthews_corr = [
                matthews_corrcoef(predictions[:, i], references[:, i], sample_weight=sample_weight)
                for i in range(references.shape[1])
            ]
            if average == "macro":
                matthews_corr = np.mean(matthews_corr)
            elif average is not None:
                raise ValueError("Invalid `average`: expected `macro`, or None ")
        else:
            matthews_corr = float(matthews_corrcoef(references, predictions, sample_weight=sample_weight))
        return {"matthews_correlation": matthews_corr}

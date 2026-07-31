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
"""Expected Calibration Error Metric"""

import numpy as np
from netcal.metrics.confidence import ECE as NetcalECE

import datasets
import evaluate


_CITATION = """\
@inproceedings{guo2017calibration,
  title={On Calibration of Modern Neural Networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle={Proceedings of the 34th International Conference on Machine Learning},
  year={2017}
}
"""

_DESCRIPTION = """\
Expected Calibration Error (ECE) measures how well a classifier's predicted
confidence scores align with actual outcomes. Predictions are binned by
confidence, and ECE is the weighted average of |accuracy - confidence| across
bins. A perfectly calibrated model has ECE = 0.
"""

_KWARGS_DESCRIPTION = """
Args:
    references : array-like of shape (n_samples,)
        Ground truth binary labels. Can be numeric (0/1 or -1/1) or strings.
    predictions : array-like of shape (n_samples,)
        Predicted confidence scores (probabilities in [0, 1]) for the positive
        class.
    n_bins : int, default=10
        Number of bins for computing the calibration error.
    equal_intervals : bool, default=True
        Binning strategy: True for equal-width bins in [0, 1],
        False for equal-mass (quantile) adaptive bins.
    pos_label : int or str, default=None
        Label of the positive class. `pos_label` will be inferred as follows:
        * if `references` in {-1, 1} or {0, 1}, `pos_label` defaults to 1;
        * else if `references` contains string, an error will be raised and
          `pos_label` should be explicitly specified;
        * otherwise, `pos_label` defaults to the greater label,
          i.e. `np.unique(references)[-1]`.

Returns:
    ece : float
        Expected Calibration Error.

Examples:
    Example-1: if references in {-1, 1} or {0, 1}, pos_label defaults to 1.
        >>> import numpy as np
        >>> ece = evaluate.load("ece")
        >>> references = np.array([0, 0, 1, 1])
        >>> predictions = np.array([0.25, 0.25, 0.75, 0.75])
        >>> results = ece.compute(references=references, predictions=predictions, n_bins=2)
        >>> print(round(results["ece"], 4))
        0.25

    Example-2: if references contains string, pos_label should be explicitly
    specified.
        >>> import numpy as np
        >>> ece = evaluate.load("ece")
        >>> references =  np.array(["spam", "ham", "ham", "spam"])
        >>> predictions = np.array([0.1, 0.9, 0.8, 0.3])
        >>> results = ece.compute(references=references, predictions=predictions, pos_label="ham")
        >>> print(round(results["ece"], 4))
        0.175
"""


def _resolve_pos_label(references):
    unique = np.unique(references)
    if unique.dtype.kind in {"U", "S", "O"}:
        raise ValueError(
            "references contains strings. pos_label must be explicitly specified."
        )
    if set(unique) <= {0, 1} or set(unique) <= {-1, 1}:
        return 1
    return int(unique[-1])


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ECE(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._get_feature_types(),
            reference_urls=["https://arxiv.org/abs/1706.04599"],
        )

    def _get_feature_types(self):
        if self.config_name == "multilist":
            return [
                datasets.Features({
                    "references": datasets.Sequence(datasets.Value("float")),
                    "predictions": datasets.Sequence(datasets.Value("float")),
                }),
                datasets.Features({
                    "references": datasets.Sequence(datasets.Value("string")),
                    "predictions": datasets.Sequence(datasets.Value("float")),
                }),
            ]
        else:
            return [
                datasets.Features({
                    "references": datasets.Value("float"),
                    "predictions": datasets.Value("float"),
                }),
                datasets.Features({
                    "references": datasets.Value("string"),
                    "predictions": datasets.Value("float"),
                }),
            ]

    def _compute(
        self,
        references,
        predictions,
        n_bins=10,
        equal_intervals=True,
        pos_label=None,
    ):
        references = np.asarray(references)
        predictions = np.asarray(predictions, dtype=np.float64)

        if pos_label is None:
            pos_label = _resolve_pos_label(references)

        labels = (references == pos_label).astype(np.float64)

        ece = NetcalECE(bins=n_bins, equal_intervals=equal_intervals)
        return {"ece": float(ece.measure(predictions, labels))}

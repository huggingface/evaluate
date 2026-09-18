# Copyright 2026 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Retrieval-Conditioned Confidence Metric (RCCS) for RAG evaluation."""

import datasets
import numpy as np
from scipy.stats import pearsonr

import evaluate


_DESCRIPTION = """
Retrieval-Conditioned Confidence Score (RCCS) evaluates the alignment of a model's confidence with its performance,
conditioned on the quality of the retrieved information.
It evaluates the joint alignment between retrieval relevance score (R), model confidence score (C), and ground-truth correctness (A).
"""


_KWARGS_DESCRIPTION = """
Args:
    retrieval_score (`list` of `float`): Retrieval relevance score (R) per example.
    confidence_score (`list` of `float`): Model confidence (C) per example. Calibrated probability recommended.
    correctness (`list` of `int`): Ground-truth correctness (A) per example (0 or 1).

Returns:
    rccs_correlation (`float`): Pearson correlation between `(R * C)` and `A`.
    confidence_calibration_error (`float`): Mean absolute error between `(R * C)` and `A`.
    mean_rc (`float`): Mean of `R * C`.
    n (`int`): Number of examples.

Examples:

    Example 1 - A simple example using lists of scores and correctness:
        >>> rccs_metric = evaluate.load("rccs")
        >>> results = rccs_metric.compute(retrieval_score=[0.8, 0.3, 0.5], confidence_score=[0.9, 0.2, 0.7], correctness=[1, 0, 1])
        >>> print(results['n'])
        3
        >>> print(round(results['mean_rc'], 2))
        0.38
        >>> print(round(results['confidence_calibration_error'], 2))
        0.33
        >>> print(round(results['rccs_correlation'], 2))
        0.83
"""


_CITATION = """
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class RCCS(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "retrieval_score": datasets.Value("float32"),
                    "confidence_score": datasets.Value("float32"),
                    "correctness": datasets.Value("int32"),
                }
            ),
            reference_urls=[],
        )

    def _compute(self, retrieval_score, confidence_score, correctness):
        R = np.array(retrieval_score, dtype=np.float32)
        C = np.array(confidence_score, dtype=np.float32)
        A = np.array(correctness, dtype=np.int32)

        if len(R) == 0:
            return {
                "rccs_correlation": float("nan"),
                "confidence_calibration_error": float("nan"),
                "mean_rc": float("nan"),
                "n": 0,
            }

        rc = R * C
        mean_rc = float(np.mean(rc))
        confidence_calibration_error = float(np.mean(np.abs(rc - A)))
        n = len(R)

        if n < 2 or np.all(rc == rc[0]) or np.all(A == A[0]):
            rccs_correlation = float("nan")
        else:
            rccs_correlation = float(pearsonr(rc, A)[0])

        return {
            "rccs_correlation": rccs_correlation,
            "confidence_calibration_error": confidence_calibration_error,
            "mean_rc": mean_rc,
            "n": n,
        }

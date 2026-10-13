# Copyright 2026 The HuggingFace Evaluate Authors.
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
import unittest

import numpy as np
from compute_score import compute_score, get_aupr, process_precisions


class TestProcessPrecisions(unittest.TestCase):
    def test_normal_case_is_unchanged(self):
        # Reversed input: [0.2, 0.5, 0.1] -> running max from the end -> [0.5, 0.5, 0.1]
        self.assertEqual(process_precisions([0.1, 0.5, 0.2]), [0.5, 0.5, 0.2])

    def test_nan_propagates_regardless_of_which_side_of_the_pair_it_is_on(self):
        # Python's built-in max() is order-dependent with NaN: max(0.5, nan)
        # is 0.5, but max(nan, 0.5) is nan. process_precisions's running-max
        # pass must not let that order dependence silently drop an undefined
        # precision - once nan enters the running max (in either argument
        # order), it must propagate through every subsequent entry. The last
        # element (highest recall) is never touched by the running max itself
        # and is unaffected either way.
        result = process_precisions([0.5, float("nan"), 0.2])
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[2], 0.2)

        result = process_precisions([0.2, float("nan"), 0.5])
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[2], 0.5)


class TestCUADAupr(unittest.TestCase):
    def test_get_aupr_normal_case_is_unchanged(self):
        # Single point: np.trapz has nothing to integrate over, so this is a
        # genuine 0.0, not an undefined integral - must stay 0.0.
        self.assertEqual(get_aupr([1.0], [1.0]), 0.0)

        # Two points spanning the full recall range at full precision: area is 1.0.
        self.assertAlmostEqual(get_aupr([1.0, 1.0], [0.0, 1.0]), 1.0)

    def test_get_aupr_propagates_nan_instead_of_flooring_to_zero(self):
        # A NaN anywhere in precisions/recalls (e.g. from a sample with zero
        # ground truths and zero predictions, which compute_precision_recall
        # scores as 0/0 -> nan) makes the integral undefined. That must be
        # reported as nan, not silently floored to 0 (the worst attainable score).
        aupr = get_aupr([1.0, float("nan"), 0.5], [0.0, 0.5, 1.0])
        self.assertTrue(np.isnan(aupr))

    def test_compute_score_reports_nan_aupr_when_a_sample_has_undefined_precision_recall(self):
        # End-to-end, with a second, normally-scored sample alongside the
        # degenerate one (a single-sample dataset is a separate edge case:
        # np.trapezoid returns 0.0 for any one-point curve regardless of its
        # value, real or nan, since there's no interval to integrate over).
        #
        # q2 has no ground-truth answers and no predicted answer text - a
        # legitimate "correctly abstained" case, but it leaves precision and
        # recall undefined (0/0) for that sample. The aggregate aupr must
        # surface as nan rather than as a perfect-worst-case 0.0.
        dataset = [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {"id": "q1", "answers": [{"text": "foo"}]},
                            {"id": "q2", "answers": []},
                        ]
                    }
                ]
            }
        ]
        predictions = {"q1": ["foo"], "q2": []}
        result = compute_score(dataset, predictions)
        self.assertTrue(np.isnan(result["aupr"]))


if __name__ == "__main__":
    unittest.main()

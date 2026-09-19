# Copyright 2025 The HuggingFace Evaluate Authors.
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

"""Tests for the label_distribution measurement."""

import math
import unittest

from label_distribution import LabelDistribution


measurement = LabelDistribution()


class TestLabelDistribution(unittest.TestCase):
    def test_uniform_binary(self):
        """Perfectly balanced binary labels should have normalized entropy of 1.0."""
        data = [0, 1, 0, 1, 0, 1]
        result = measurement.compute(data=data)
        self.assertAlmostEqual(result["label_entropy_normalized"], 1.0)
        self.assertAlmostEqual(result["label_entropy"], math.log(2))
        self.assertEqual(result["label_distribution"]["fractions"], [0.5, 0.5])

    def test_uniform_multiclass(self):
        """Perfectly balanced 3-class labels should have normalized entropy of 1.0."""
        data = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        result = measurement.compute(data=data)
        self.assertAlmostEqual(result["label_entropy_normalized"], 1.0)
        self.assertAlmostEqual(result["label_entropy"], math.log(3))

    def test_single_class(self):
        """All labels the same should have entropy 0."""
        data = [1, 1, 1, 1, 1]
        result = measurement.compute(data=data)
        self.assertAlmostEqual(result["label_entropy"], 0.0)
        self.assertAlmostEqual(result["label_entropy_normalized"], 0.0)

    def test_imbalanced(self):
        """Imbalanced labels should have normalized entropy less than 1."""
        data = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        result = measurement.compute(data=data)
        self.assertGreater(result["label_entropy"], 0.0)
        self.assertLess(result["label_entropy_normalized"], 1.0)

    def test_permutation_invariance(self):
        """Entropy should be the same regardless of which integer is assigned to which class.

        This is the key property that skewness lacked: [0,0,1,1,1,1,1,2,2] and
        [0,0,1,1,2,2,2,2,2] have the same class distribution (2,5,2) but
        different skewness.  Entropy must be identical for both.
        """
        data_a = [0, 0, 1, 1, 1, 1, 1, 2, 2]
        data_b = [0, 0, 1, 1, 2, 2, 2, 2, 2]
        result_a = measurement.compute(data=data_a)
        result_b = measurement.compute(data=data_b)
        self.assertAlmostEqual(result_a["label_entropy"], result_b["label_entropy"])
        self.assertAlmostEqual(result_a["label_entropy_normalized"], result_b["label_entropy_normalized"])

    def test_string_labels(self):
        """String labels should work the same as integer labels."""
        data = ["cat", "dog", "cat", "cat", "dog"]
        result = measurement.compute(data=data)
        self.assertGreater(result["label_entropy"], 0.0)
        self.assertLess(result["label_entropy_normalized"], 1.0)
        self.assertIn("cat", result["label_distribution"]["labels"])
        self.assertIn("dog", result["label_distribution"]["labels"])

    def test_output_keys(self):
        """Output should contain label_distribution, label_entropy, and label_entropy_normalized."""
        data = [0, 1, 2]
        result = measurement.compute(data=data)
        self.assertIn("label_distribution", result)
        self.assertIn("label_entropy", result)
        self.assertIn("label_entropy_normalized", result)
        self.assertIn("labels", result["label_distribution"])
        self.assertIn("fractions", result["label_distribution"])

    def test_fractions_sum_to_one(self):
        """Label fractions should always sum to 1."""
        data = [0, 0, 1, 2, 2, 2, 3]
        result = measurement.compute(data=data)
        self.assertAlmostEqual(sum(result["label_distribution"]["fractions"]), 1.0)


if __name__ == "__main__":
    unittest.main()

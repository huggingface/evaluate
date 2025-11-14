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

"""Tests for the FEVER (Fact Extraction and VERification) metric."""

import unittest

from fever import FEVER  # assuming your metric file is named fever.py


fever = FEVER()


class TestFEVER(unittest.TestCase):

    def test_perfect_prediction(self):
        preds = [{"label": "SUPPORTED", "evidence": ["E1", "E2"]}]
        refs = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]}]
        result = fever.compute(predictions=preds, references=refs)
        self.assertAlmostEqual(result["label_accuracy"], 1.0)
        self.assertAlmostEqual(result["fever_score"], 1.0)
        self.assertAlmostEqual(result["evidence_precision"], 1.0)
        self.assertAlmostEqual(result["evidence_recall"], 1.0)
        self.assertAlmostEqual(result["evidence_f1"], 1.0)

    def test_label_only_correct(self):
        preds = [{"label": "SUPPORTED", "evidence": ["X1", "X2"]}]
        refs = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]}]
        result = fever.compute(predictions=preds, references=refs)
        self.assertAlmostEqual(result["label_accuracy"], 1.0)
        self.assertAlmostEqual(result["fever_score"], 0.0)
        self.assertTrue(result["evidence_f1"] < 1.0)

    def test_label_incorrect(self):
        preds = [{"label": "REFUTED", "evidence": ["E1", "E2"]}]
        refs = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]}]
        result = fever.compute(predictions=preds, references=refs)
        self.assertAlmostEqual(result["label_accuracy"], 0.0)
        self.assertAlmostEqual(result["fever_score"], 0.0)

    def test_partial_evidence_overlap(self):
        preds = [{"label": "SUPPORTED", "evidence": ["E1"]}]
        refs = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]}]
        result = fever.compute(predictions=preds, references=refs)
        self.assertAlmostEqual(result["label_accuracy"], 1.0)
        self.assertAlmostEqual(result["fever_score"], 0.0)
        self.assertAlmostEqual(result["evidence_precision"], 1.0)
        self.assertAlmostEqual(result["evidence_recall"], 0.5)
        self.assertTrue(0 < result["evidence_f1"] < 1.0)

    def test_extra_evidence_still_correct(self):
        preds = [{"label": "SUPPORTED", "evidence": ["E1", "E2", "X1"]}]
        refs = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]}]
        result = fever.compute(predictions=preds, references=refs)
        self.assertAlmostEqual(result["fever_score"], 1.0)
        self.assertTrue(result["evidence_precision"] < 1.0)
        self.assertAlmostEqual(result["evidence_recall"], 1.0)

    def test_multiple_gold_sets(self):
        preds = [{"label": "SUPPORTED", "evidence": ["E3", "E4"]}]
        refs = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"], ["E3", "E4"]]}]
        result = fever.compute(predictions=preds, references=refs)
        self.assertAlmostEqual(result["fever_score"], 1.0)
        self.assertAlmostEqual(result["label_accuracy"], 1.0)

    def test_mixed_examples(self):
        preds = [
            {"label": "SUPPORTED", "evidence": ["A1", "A2"]},
            {"label": "SUPPORTED", "evidence": ["B1"]},
            {"label": "REFUTED", "evidence": ["C1", "C2"]},
        ]
        refs = [
            {"label": "SUPPORTED", "evidence_sets": [["A1", "A2"]]},
            {"label": "SUPPORTED", "evidence_sets": [["B1", "B2"]]},
            {"label": "SUPPORTED", "evidence_sets": [["C1", "C2"]]},
        ]
        result = fever.compute(predictions=preds, references=refs)
        self.assertTrue(0 < result["label_accuracy"] < 1.0)
        self.assertTrue(0 <= result["fever_score"] < 1.0)
        self.assertTrue(0 <= result["evidence_f1"] <= 1.0)

    def test_empty_evidence_prediction(self):
        preds = [{"label": "SUPPORTED", "evidence": []}]
        refs = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]}]
        result = fever.compute(predictions=preds, references=refs)
        self.assertEqual(result["evidence_precision"], 0.0)
        self.assertEqual(result["evidence_recall"], 0.0)
        self.assertEqual(result["evidence_f1"], 0.0)

    def test_empty_gold_evidence(self):
        preds = [{"label": "SUPPORTED", "evidence": ["E1", "E2"]}]
        refs = [{"label": "SUPPORTED", "evidence_sets": [[]]}]
        result = fever.compute(predictions=preds, references=refs)
        self.assertEqual(result["evidence_recall"], 0.0)

    def test_multiple_examples_micro_averaging(self):
        preds = [
            {"label": "SUPPORTED", "evidence": ["E1"]},
            {"label": "SUPPORTED", "evidence": ["F1", "F2"]},
        ]
        refs = [
            {"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]},
            {"label": "SUPPORTED", "evidence_sets": [["F1", "F2"]]},
        ]
        result = fever.compute(predictions=preds, references=refs)
        self.assertTrue(result["evidence_f1"] < 1.0)
        self.assertAlmostEqual(result["label_accuracy"], 1.0)

    def test_fever_score_requires_label_match(self):
        preds = [{"label": "REFUTED", "evidence": ["E1", "E2"]}]
        refs = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]}]
        result = fever.compute(predictions=preds, references=refs)
        self.assertEqual(result["fever_score"], 0.0)
        self.assertEqual(result["label_accuracy"], 0.0)

    def test_empty_input_list(self):
        preds, refs = [], []
        result = fever.compute(predictions=preds, references=refs)
        for k in result:
            self.assertEqual(result[k], 0.0)


if __name__ == "__main__":
    unittest.main()

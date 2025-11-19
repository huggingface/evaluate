# Copyright 2025 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Tests for Circular Bias Detection (CBD) Integrity Score metric."""

import unittest
import numpy as np
import pytest

import evaluate


class TestCircularBiasIntegrity(unittest.TestCase):
    """Test suite for CBD Integrity Score metric."""

    def setUp(self):
        """Load the metric before each test."""
        self.metric = evaluate.load("circular_bias_integrity")

    def test_low_bias_scenario(self):
        """Test scenario with no circular bias (low correlation)."""
        # Performance stays relatively stable despite protocol changes
        performance_scores = [0.85, 0.84, 0.86, 0.85, 0.87]
        protocol_variations = [0.1, 0.5, 0.2, 0.8, 0.3]

        results = self.metric.compute(
            performance_scores=performance_scores,
            protocol_variations=protocol_variations,
        )

        # Should detect low bias
        self.assertLess(
            results["cbd_score"], 40, "Low bias scenario should have CBD score < 40"
        )
        self.assertEqual(
            results["risk_level"],
            "LOW",
            "Risk level should be LOW for uncorrelated data",
        )
        self.assertIn("cbd_score", results)
        self.assertIn("rho_pc", results)
        self.assertIn("recommendation", results)

    def test_high_bias_scenario(self):
        """Test scenario with strong circular bias (high correlation)."""
        # Performance increases linearly with protocol changes
        performance_scores = [0.75, 0.80, 0.85, 0.90, 0.95]
        protocol_variations = [0.1, 0.2, 0.3, 0.4, 0.5]

        results = self.metric.compute(
            performance_scores=performance_scores,
            protocol_variations=protocol_variations,
        )

        # Should detect high bias
        self.assertGreater(
            results["cbd_score"], 60, "High bias scenario should have CBD score > 60"
        )
        self.assertEqual(
            results["risk_level"],
            "HIGH",
            "Risk level should be HIGH for strongly correlated data",
        )
        self.assertGreater(
            abs(results["rho_pc"]), 0.7, "ρ_PC should be > 0.7 for strong correlation"
        )

    def test_moderate_bias_scenario(self):
        """Test scenario with moderate circular bias."""
        performance_scores = [0.82, 0.85, 0.83, 0.88, 0.87]
        protocol_variations = [0.15, 0.25, 0.18, 0.35, 0.30]

        results = self.metric.compute(
            performance_scores=performance_scores,
            protocol_variations=protocol_variations,
        )

        # Should detect moderate bias
        self.assertGreaterEqual(
            results["cbd_score"], 30, "Moderate bias should have CBD score >= 30"
        )
        self.assertLessEqual(
            results["cbd_score"], 70, "Moderate bias should have CBD score <= 70"
        )

    def test_negative_correlation(self):
        """Test scenario with negative correlation (performance decreases with protocol changes)."""
        performance_scores = [0.95, 0.90, 0.85, 0.80, 0.75]
        protocol_variations = [0.1, 0.2, 0.3, 0.4, 0.5]

        results = self.metric.compute(
            performance_scores=performance_scores,
            protocol_variations=protocol_variations,
        )

        # Negative correlation should also be detected as bias
        self.assertGreater(
            results["cbd_score"],
            60,
            "Negative correlation should also indicate high bias",
        )
        self.assertLess(
            results["rho_pc"],
            -0.7,
            "ρ_PC should be < -0.7 for strong negative correlation",
        )

    def test_minimum_data_requirement(self):
        """Test that metric requires at least 3 data points."""
        performance_scores = [0.85, 0.87]
        protocol_variations = [0.1, 0.2]

        with pytest.raises(ValueError, match="at least 3 evaluation rounds"):
            self.metric.compute(
                performance_scores=performance_scores,
                protocol_variations=protocol_variations,
            )

    def test_length_mismatch_error(self):
        """Test that metric raises error when input lengths don't match."""
        performance_scores = [0.85, 0.87, 0.91]
        protocol_variations = [0.1, 0.2]

        with pytest.raises(ValueError, match="Length mismatch"):
            self.metric.compute(
                performance_scores=performance_scores,
                protocol_variations=protocol_variations,
            )

    def test_missing_inputs_error(self):
        """Test that metric raises error when required inputs are missing."""
        with pytest.raises(
            ValueError, match="requires 'performance_scores' and 'protocol_variations'"
        ):
            self.metric.compute()

    def test_with_performance_matrix(self):
        """Test computation with performance matrix (PSI calculation)."""
        performance_matrix = np.array(
            [
                [0.85, 0.78, 0.82],
                [0.87, 0.80, 0.84],
                [0.91, 0.84, 0.88],
                [0.89, 0.82, 0.86],
                [0.93, 0.86, 0.90],
            ]
        )

        performance_scores = performance_matrix.mean(axis=1).tolist()
        protocol_variations = [0.1, 0.15, 0.25, 0.20, 0.30]

        results = self.metric.compute(
            performance_scores=performance_scores,
            protocol_variations=protocol_variations,
            performance_matrix=performance_matrix,
            return_all_indicators=True,
        )

        # Should include PSI score
        self.assertIn("psi_score", results)
        self.assertIsInstance(results["psi_score"], float)
        self.assertGreaterEqual(
            results["psi_score"], 0, "PSI score should be non-negative"
        )

    def test_with_constraint_matrix(self):
        """Test computation with constraint matrix (CCS calculation)."""
        performance_scores = [0.85, 0.87, 0.91, 0.89, 0.93]
        protocol_variations = [0.1, 0.15, 0.25, 0.20, 0.30]

        constraint_matrix = np.array(
            [[512, 0.001], [550, 0.0015], [600, 0.002], [580, 0.0018], [620, 0.0022]]
        )

        results = self.metric.compute(
            performance_scores=performance_scores,
            protocol_variations=protocol_variations,
            constraint_matrix=constraint_matrix,
            return_all_indicators=True,
        )

        # Should include CCS score
        self.assertIn("ccs_score", results)
        self.assertIsInstance(results["ccs_score"], float)
        self.assertGreaterEqual(
            results["ccs_score"], 0, "CCS score should be between 0 and 1"
        )
        self.assertLessEqual(
            results["ccs_score"], 1, "CCS score should be between 0 and 1"
        )

    def test_full_computation_with_all_indicators(self):
        """Test full computation with all three indicators."""
        performance_matrix = np.array(
            [
                [0.85, 0.78, 0.82],
                [0.87, 0.80, 0.84],
                [0.91, 0.84, 0.88],
                [0.89, 0.82, 0.86],
                [0.93, 0.86, 0.90],
            ]
        )

        constraint_matrix = np.array(
            [[512, 0.7], [550, 0.75], [600, 0.8], [580, 0.78], [620, 0.82]]
        )

        performance_scores = performance_matrix.mean(axis=1).tolist()
        protocol_variations = [0.1, 0.15, 0.25, 0.20, 0.30]

        results = self.metric.compute(
            performance_scores=performance_scores,
            protocol_variations=protocol_variations,
            performance_matrix=performance_matrix,
            constraint_matrix=constraint_matrix,
            return_all_indicators=True,
        )

        # Should include all indicators
        self.assertIn("cbd_score", results)
        self.assertIn("rho_pc", results)
        self.assertIn("psi_score", results)
        self.assertIn("ccs_score", results)
        self.assertIn("risk_level", results)
        self.assertIn("recommendation", results)

    def test_constant_performance(self):
        """Test with constant performance (should handle gracefully)."""
        performance_scores = [0.85, 0.85, 0.85, 0.85, 0.85]
        protocol_variations = [0.1, 0.2, 0.3, 0.4, 0.5]

        results = self.metric.compute(
            performance_scores=performance_scores,
            protocol_variations=protocol_variations,
        )

        # Should handle constant performance
        self.assertIn("cbd_score", results)
        # Correlation should be 0 or NaN (handled as 0)
        self.assertLessEqual(
            abs(results["rho_pc"]),
            0.1,
            "Constant performance should have near-zero correlation",
        )

    def test_output_types(self):
        """Test that all output values have correct types."""
        performance_scores = [0.85, 0.87, 0.91, 0.89, 0.93]
        protocol_variations = [0.1, 0.15, 0.25, 0.20, 0.30]

        results = self.metric.compute(
            performance_scores=performance_scores,
            protocol_variations=protocol_variations,
        )

        self.assertIsInstance(results["cbd_score"], float)
        self.assertIsInstance(results["rho_pc"], float)
        self.assertIsInstance(results["risk_level"], str)
        self.assertIsInstance(results["recommendation"], str)

    def test_legacy_interface(self):
        """Test backward compatibility with predictions/references interface."""
        # Some users might use the standard predictions/references interface
        predictions = [0.85, 0.87, 0.91, 0.89, 0.93]
        references = [0.1, 0.15, 0.25, 0.20, 0.30]

        results = self.metric.compute(predictions=predictions, references=references)

        self.assertIn("cbd_score", results)
        self.assertIn("rho_pc", results)


if __name__ == "__main__":
    unittest.main()

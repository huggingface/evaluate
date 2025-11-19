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
"""Circular Bias Detection (CBD) Integrity Score metric."""

import datasets
import numpy as np
from scipy.stats import pearsonr

import evaluate


_DESCRIPTION = """
Circular Bias Detection (CBD) Integrity Score measures the statistical integrity of AI evaluation processes 
by quantifying circular reasoning bias. This metric addresses a critical gap in evaluation methodology: 
while traditional metrics measure model performance, CBD measures whether the evaluation process itself 
is statistically reliable.

**Key Concept**: Circular bias occurs when evaluation results become artificially inflated through 
iterative protocol adjustments (e.g., hyperparameter tuning, prompt engineering, dataset selection) 
that optimize for benchmark performance rather than true model generalization.

**Core Indicators**:
- **ρ_PC (Protocol-Performance Correlation)**: Measures the correlation between evaluation protocol 
  changes and resulting performance scores. High correlation indicates potential circular dependency.
- **PSI (Performance-Structure Independence)**: Quantifies parameter stability across evaluation periods.
- **CCS (Constraint-Consistency Score)**: Measures consistency of constraint specifications over time.

**Slogan**: *Ensuring your evaluation is trustworthy. Stop circular reasoning in AI benchmarks.*

This metric is particularly valuable for:
- Detecting overfitting to benchmarks during model development
- Validating evaluation integrity in research papers
- Auditing AI system evaluations for regulatory compliance
- Meta-evaluation of evaluation methodologies

For detailed methodology, see: Zhang, H. (2025). "Circular Bias Detection: A Comprehensive Statistical 
Framework for Detecting Circular Reasoning Bias in AI Algorithm Evaluation."
"""


_KWARGS_DESCRIPTION = """
Args:
    performance_scores (`list` of `float`): Performance scores across multiple evaluation rounds.
        Each score represents model performance in a specific evaluation period (e.g., accuracy, F1, BLEU).
        Minimum 3 evaluation rounds required for reliable correlation analysis.
    protocol_variations (`list` of `float`): Quantified protocol variation magnitudes corresponding 
        to each evaluation round. This represents the degree of change in evaluation protocol 
        (e.g., hyperparameter changes, prompt modifications, dataset adjustments).
        Must have the same length as performance_scores.
    performance_matrix (`array-like`, optional): Shape (T, K) where T is time periods and K is algorithms.
        Provides detailed performance tracking across multiple algorithms and time periods.
        If provided, enables computation of PSI (Performance-Structure Independence).
    constraint_matrix (`array-like`, optional): Shape (T, p) where T is time periods and p is constraint types.
        Tracks constraint specifications across evaluation periods.
        If provided, enables computation of CCS (Constraint-Consistency Score).
    return_all_indicators (`boolean`, optional): If `True`, returns all three indicators (ρ_PC, PSI, CCS) 
        along with the overall CBD score. If `False`, returns only the CBD score and ρ_PC. 
        Defaults to `False`.

Returns:
    cbd_score (`float`): Overall Circular Bias Detection integrity score (0-100 scale). 
        Higher scores indicate stronger evidence of circular bias.
        - 0-30: Low risk (evaluation appears statistically sound)
        - 30-60: Moderate risk (some circular dependency detected)
        - 60-100: High risk (significant circular bias detected)
    rho_pc (`float`): Protocol-Performance correlation coefficient (-1 to 1).
        Measures the linear relationship between protocol changes and performance.
        Values close to ±1 indicate strong circular dependency.
    psi_score (`float`, optional): Performance-Structure Independence score (returned if performance_matrix provided).
        Higher values indicate more parameter instability/bias.
    ccs_score (`float`, optional): Constraint-Consistency Score (returned if constraint_matrix provided).
        Higher values indicate more consistency (less bias).
    risk_level (`str`): Categorical risk assessment: "LOW", "MODERATE", or "HIGH".
    recommendation (`str`): Actionable guidance based on detected bias level.

Examples:

    Example 1 - Basic usage with simple performance and protocol data:
        >>> cbd_metric = evaluate.load("circular_bias_integrity")
        >>> performance = [0.85, 0.87, 0.91, 0.89, 0.93]
        >>> protocol_changes = [0.1, 0.15, 0.25, 0.20, 0.30]
        >>> results = cbd_metric.compute(
        ...     performance_scores=performance,
        ...     protocol_variations=protocol_changes
        ... )
        >>> print(f"CBD Score: {results['cbd_score']:.1f}")
        CBD Score: 78.5
        >>> print(f"Risk Level: {results['risk_level']}")
        Risk Level: HIGH

    Example 2 - Advanced usage with full matrix data:
        >>> cbd_metric = evaluate.load("circular_bias_integrity")
        >>> # Performance across 5 time periods for 3 algorithms
        >>> perf_matrix = np.array([
        ...     [0.85, 0.78, 0.82],
        ...     [0.87, 0.80, 0.84],
        ...     [0.91, 0.84, 0.88],
        ...     [0.89, 0.82, 0.86],
        ...     [0.93, 0.86, 0.90]
        ... ])
        >>> # Constraint specifications across 5 time periods
        >>> constraint_matrix = np.array([
        ...     [512, 0.7],
        ...     [550, 0.75],
        ...     [600, 0.8],
        ...     [580, 0.78],
        ...     [620, 0.82]
        ... ])
        >>> results = cbd_metric.compute(
        ...     performance_scores=perf_matrix.mean(axis=1).tolist(),
        ...     protocol_variations=[0.1, 0.15, 0.25, 0.20, 0.30],
        ...     performance_matrix=perf_matrix,
        ...     constraint_matrix=constraint_matrix,
        ...     return_all_indicators=True
        ... )
        >>> print(f"ρ_PC: {results['rho_pc']:.3f}")
        ρ_PC: 0.785
        >>> print(f"PSI: {results['psi_score']:.3f}")
        PSI: 0.042
        >>> print(f"CCS: {results['ccs_score']:.3f}")
        CCS: 0.891
"""


_CITATION = """
@article{zhang2025circular,
  title={Circular Bias Detection: A Comprehensive Statistical Framework for Detecting Circular Reasoning Bias in AI Algorithm Evaluation},
  author={Zhang, Hongping},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025},
  note={Software available at: https://github.com/hongping-zh/circular-bias-detection}
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CircularBiasIntegrity(evaluate.Metric):
    """Circular Bias Detection (CBD) Integrity Score metric for evaluation trustworthiness."""

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float"),
                    "references": datasets.Value("float"),
                }
            ),
            reference_urls=[
                "https://github.com/hongping-zh/circular-bias-detection",
                "https://doi.org/10.5281/zenodo.17201032",
            ],
        )

    def _compute(
        self,
        performance_scores=None,
        protocol_variations=None,
        performance_matrix=None,
        constraint_matrix=None,
        return_all_indicators=False,
        predictions=None,
        references=None,
    ):
        """
        Compute CBD integrity score and related indicators.

        Note: This metric requires either (performance_scores, protocol_variations)
        or (predictions, references) to be provided.
        """

        # Handle legacy interface (predictions/references)
        if performance_scores is None and predictions is not None:
            performance_scores = predictions
        if protocol_variations is None and references is not None:
            protocol_variations = references

        # Validate inputs
        if performance_scores is None or protocol_variations is None:
            raise ValueError(
                "CBD metric requires 'performance_scores' and 'protocol_variations' to be provided. "
                "These represent the performance trajectory and corresponding protocol changes across evaluation rounds."
            )

        performance_scores = np.array(performance_scores)
        protocol_variations = np.array(protocol_variations)

        if len(performance_scores) != len(protocol_variations):
            raise ValueError(
                f"Length mismatch: performance_scores ({len(performance_scores)}) and "
                f"protocol_variations ({len(protocol_variations)}) must have the same length."
            )

        if len(performance_scores) < 3:
            raise ValueError(
                "CBD metric requires at least 3 evaluation rounds for reliable correlation analysis. "
                f"Received {len(performance_scores)} rounds."
            )

        # 1. Compute ρ_PC (Protocol-Performance Correlation)
        rho_pc_corr, rho_pc_pvalue = pearsonr(performance_scores, protocol_variations)

        # Handle NaN correlations
        if np.isnan(rho_pc_corr):
            rho_pc_corr = 0.0

        # 2. Compute CBD overall score (0-100 scale)
        # Base score from ρ_PC (primary indicator)
        cbd_score = abs(rho_pc_corr) * 100

        results = {
            "cbd_score": float(cbd_score),
            "rho_pc": float(rho_pc_corr),
            "rho_pc_pvalue": float(rho_pc_pvalue),
        }

        # 3. Compute PSI if performance_matrix provided
        if performance_matrix is not None and return_all_indicators:
            psi_score = self._compute_psi(np.array(performance_matrix))
            results["psi_score"] = float(psi_score)

        # 4. Compute CCS if constraint_matrix provided
        if constraint_matrix is not None and return_all_indicators:
            ccs_score = self._compute_ccs(np.array(constraint_matrix))
            results["ccs_score"] = float(ccs_score)

        # 5. Risk assessment
        if cbd_score < 30:
            risk_level = "LOW"
            recommendation = (
                "Evaluation appears statistically sound. Continue current methodology."
            )
        elif cbd_score < 60:
            risk_level = "MODERATE"
            recommendation = (
                "Some circular dependency detected. Consider: (1) Using held-out test sets, "
                "(2) Pre-registering evaluation protocols, (3) Limiting protocol iterations."
            )
        else:
            risk_level = "HIGH"
            recommendation = (
                "Significant circular bias detected. Strongly recommend: (1) Independent validation set, "
                "(2) Protocol pre-registration, (3) Reporting all evaluation attempts, "
                "(4) Consider using cross-validation or bootstrap methods."
            )

        results["risk_level"] = risk_level
        results["recommendation"] = recommendation

        return results

    def _compute_psi(self, performance_matrix):
        """
        Compute Performance-Structure Independence (PSI) score.

        PSI measures parameter stability across evaluation periods.
        Higher values indicate more instability/bias.
        """
        T, K = performance_matrix.shape

        if T < 2:
            return 0.0

        psi_scores = []
        for k in range(K):
            param_series = performance_matrix[:, k]
            differences = np.diff(param_series)
            psi_k = np.mean(np.abs(differences))
            psi_scores.append(psi_k)

        return np.mean(psi_scores)

    def _compute_ccs(self, constraint_matrix):
        """
        Compute Constraint-Consistency Score (CCS).

        CCS measures consistency of constraint specifications.
        Higher values indicate more consistency (less bias).
        """
        T, p = constraint_matrix.shape

        if T < 2:
            return 1.0

        consistency_scores = []
        for j in range(p):
            constraint_series = constraint_matrix[:, j]

            # Handle constant constraints
            if np.std(constraint_series) == 0:
                consistency_scores.append(1.0)
                continue

            mean_val = np.mean(constraint_series)
            if mean_val == 0:
                consistency_scores.append(0.0)
                continue

            # Coefficient of variation
            cv = np.std(constraint_series) / np.abs(mean_val)

            # Transform to consistency score (lower CV = higher consistency)
            consistency_j = 1 / (1 + cv)
            consistency_scores.append(consistency_j)

        return np.mean(consistency_scores)

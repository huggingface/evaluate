from typing import Any, Dict, List, Optional

import datasets
import evaluate
import numpy as np


_DESCRIPTION = """
Human Trust & Uncertainty Metrics for AI Evaluation.

This metric suite operationalizes trust calibration, belief updating,
and uncertainty alignment for human–AI interaction evaluation.
It complements traditional performance metrics by surfacing
human-centered signals about trust, belief dynamics, and confidence communication.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (List[Any]):
        Model predictions.
    references (List[Any]):
        Ground truth labels.
    confidences (List[float]):
        Model confidence values in [0, 1].
    human_trust_scores (List[float]):
        Human trust ratings in [0, 1].
    belief_priors (Optional[List[float]]):
        User beliefs before seeing AI output.
    belief_posteriors (Optional[List[float]]):
        User beliefs after seeing AI output.
    explanation_complexity (Optional[List[float]]):
        Explanation complexity scores (e.g., length, entropy, readability).

Returns:
    Dict[str, float]:
        A dictionary containing:
            - expected_trust_error
            - trust_sensitivity_index
            - belief_shift_magnitude (optional)
            - overconfidence_penalty
            - overconfidence_penalty_normalized
            - explanation_confidence_alignment (optional)
"""


def _safe_mean(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.mean(x))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if np.isnan(corr) else corr


class HumanAITrust(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                    "confidences": datasets.Value("float32"),
                    "human_trust_scores": datasets.Value("float32"),
                }
            ),
            reference_urls=[],
        )

    def _validate_inputs(
        self,
        predictions: List[Any],
        references: List[Any],
        confidences: List[float],
        human_trust_scores: List[float],
        belief_priors: Optional[List[float]],
        belief_posteriors: Optional[List[float]],
        explanation_complexity: Optional[List[float]]
    ) -> None:
        # Length checks
        n = len(predictions)
        if not (
            len(references) == n and
            len(confidences) == n and
            len(human_trust_scores) == n
        ):
            raise ValueError("All required input lists must have equal length.")

        if belief_priors is not None and len(belief_priors) != n:
            raise ValueError("belief_priors must have the same length as predictions.")

        if belief_posteriors is not None and len(belief_posteriors) != n:
            raise ValueError("belief_posteriors must have the same length as predictions.")

        if explanation_complexity is not None and len(explanation_complexity) != n:
            raise ValueError("explanation_complexity must have the same length as predictions.")

        # Range checks
        for c in confidences:
            if not (0.0 <= c <= 1.0):
                raise ValueError("All confidence values must be in [0, 1].")

        for t in human_trust_scores:
            if not (0.0 <= t <= 1.0):
                raise ValueError("All human trust scores must be in [0, 1].")

        if belief_priors is not None:
            for b in belief_priors:
                if not (0.0 <= b <= 1.0):
                    raise ValueError("All belief_priors values must be in [0, 1].")

        if belief_posteriors is not None:
            for b in belief_posteriors:
                if not (0.0 <= b <= 1.0):
                    raise ValueError("All belief_posteriors values must be in [0, 1].")

    def _compute(
        self,
        predictions: List[Any],
        references: List[Any],
        confidences: List[float],
        human_trust_scores: List[float],
        belief_priors: Optional[List[float]] = None,
        belief_posteriors: Optional[List[float]] = None,
        explanation_complexity: Optional[List[float]] = None,
    ) -> Dict[str, Optional[float]]:

        # Input validation
        self._validate_inputs(
            predictions,
            references,
            confidences,
            human_trust_scores,
            belief_priors,
            belief_posteriors,
            explanation_complexity,
        )

        # Convert to numpy
        confidences = np.array(confidences, dtype=float)
        trust = np.array(human_trust_scores, dtype=float)

        # === Expected Trust Error (ETE) ===
        ete = _safe_mean(np.abs(trust - confidences))

        # === Trust Sensitivity Index (TSI) ===
        tsi = _safe_corr(trust, confidences)

        # === Belief Shift Magnitude (BSM) ===
        if belief_priors is not None and belief_posteriors is not None:
            belief_priors_arr = np.array(belief_priors, dtype=float)
            belief_posteriors_arr = np.array(belief_posteriors, dtype=float)
            bsm = _safe_mean(np.abs(belief_posteriors_arr - belief_priors_arr))
        else:
            bsm = None

        # === Explanation–Confidence Alignment (ECA) ===
        if explanation_complexity is not None:
            expl = np.array(explanation_complexity, dtype=float)
            eca = _safe_corr(confidences, expl)
        else:
            eca = None

        # === Overconfidence Penalty (OCP) ===
        errors = np.array(
            [pred != ref for pred, ref in zip(predictions, references)],
            dtype=float
        )
        ocp = _safe_mean(confidences * errors)

        # === Normalized Overconfidence Penalty ===
        mean_conf = _safe_mean(confidences)
        if mean_conf > 0:
            ocp_norm = ocp / mean_conf
        else:
            ocp_norm = 0.0

        results = {
            "expected_trust_error": float(ete),
            "trust_sensitivity_index": float(tsi),
            "belief_shift_magnitude": None if bsm is None else float(bsm),
            "overconfidence_penalty": float(ocp),
            "overconfidence_penalty_normalized": float(ocp_norm),
            "explanation_confidence_alignment": None if eca is None else float(eca),
        }

        return results
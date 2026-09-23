import evaluate
import numpy as np


def test_basic_metrics():
    metric = evaluate.load("human_ai_trust")

    preds = [1, 0, 1, 1]
    refs = [1, 1, 0, 1]
    confs = [0.9, 0.7, 0.8, 0.6]
    trust = [0.85, 0.6, 0.75, 0.65]

    out = metric.compute(
        predictions=preds,
        references=refs,
        confidences=confs,
        human_trust_scores=trust,
    )

    # ETE
    ete_expected = np.mean(np.abs(np.array(trust) - np.array(confs)))
    assert abs(out["expected_trust_error"] - ete_expected) < 1e-6

    # TSI
    tsi_expected = np.corrcoef(trust, confs)[0, 1]
    assert abs(out["trust_sensitivity_index"] - tsi_expected) < 1e-6

    # OCP
    errors = np.array([p != r for p, r in zip(preds, refs)], dtype=float)
    ocp_expected = np.mean(np.array(confs) * errors)
    assert abs(out["overconfidence_penalty"] - ocp_expected) < 1e-6

    # OCP_norm
    ocp_norm_expected = ocp_expected / np.mean(confs)
    assert abs(out["overconfidence_penalty_normalized"] - ocp_norm_expected) < 1e-6


def test_zero_variance_confidence():
    metric = evaluate.load("human_ai_trust")

    preds = [1, 0, 1]
    refs = [1, 1, 0]
    confs = [0.5, 0.5, 0.5]
    trust = [0.4, 0.6, 0.5]

    out = metric.compute(
        predictions=preds,
        references=refs,
        confidences=confs,
        human_trust_scores=trust,
    )

    assert out["trust_sensitivity_index"] == 0.0


def test_bsm_and_eca():
    metric = evaluate.load("human_ai_trust")

    preds = [1, 0, 1]
    refs = [1, 1, 0]
    confs = [0.9, 0.7, 0.8]
    trust = [0.85, 0.6, 0.75]
    priors = [0.3, 0.4, 0.5]
    posts = [0.6, 0.5, 0.7]
    expl = [10, 20, 15]

    out = metric.compute(
        predictions=preds,
        references=refs,
        confidences=confs,
        human_trust_scores=trust,
        belief_priors=priors,
        belief_posteriors=posts,
        explanation_complexity=expl,
    )

    bsm_expected = np.mean(np.abs(np.array(posts) - np.array(priors)))
    assert abs(out["belief_shift_magnitude"] - bsm_expected) < 1e-6

    eca_expected = np.corrcoef(confs, expl)[0, 1]
    assert abs(out["explanation_confidence_alignment"] - eca_expected) < 1e-6


def test_missing_optional_inputs():
    metric = evaluate.load("human_ai_trust")

    preds = [1, 0]
    refs = [1, 1]
    confs = [0.8, 0.6]
    trust = [0.75, 0.65]

    out = metric.compute(
        predictions=preds,
        references=refs,
        confidences=confs,
        human_trust_scores=trust,
    )

    assert out["belief_shift_magnitude"] is None
    assert out["explanation_confidence_alignment"] is None
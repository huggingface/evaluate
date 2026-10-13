"""Behavioral Drift metric for HuggingFace Evaluate.

Detects fine-tuning output collapse that complements perplexity.
Concatenated-reference self-BLEU | digit density | repetition ratio
=> drift_score (0=collapse, 1=healthy)

Motivated by observations from LoRA fine-tuning where training loss
improved while outputs collapsed to repetitive patterns.
"""
import evaluate
import datasets
from collections import Counter

_CITATION = """\
@misc{behavioral_drift,
  author = {Yuhao Lin},
  title = {Behavioral Drift: A Fine-Tuning Quality Metric},
  year = {2026},
}
"""

_DESCRIPTION = """\
Behavioral Drift catches fine-tuning output collapse earlier than perplexity alone.

Three signals composited into a 0-1 drift_score:
1. self-BLEU: concatenated-reference BLEU-1 across outputs (>0.8 = mode collapse)
2. digit_density: fraction of numeric chars, vs reference baseline
3. repetition_ratio: unique/total token ratio (<0.4 = degenerate loops)

Real-world case: Qwen2.5-1.5B-Instruct fine-tuned on 80 custom samples.
Loss: 9.2 -> 8.8 (improvement). But all outputs collapsed to "1999999...".
Drift_score caught this while perplexity did not.

Unified thresholds across formula and diagnosis:
self-BLEU>0.8, digit_density_delta>0.2, repetition_ratio<0.4
"""

_KWARGS_DESCRIPTION = """\
Args:
    predictions: list of fine-tuned model outputs on test prompts.
    references: list of base model outputs on same prompts (baseline).
Returns:
    drift_score (0-1), self_bleu, digit_density, digit_density_baseline,
    repetition_ratio, repetition_ratio_baseline, diagnosis
"""

# Unified thresholds — same in drift_score formula and diagnosis
SB_T = 0.8        # self-BLEU threshold
DD_DELTA_T = 0.2  # digit_density increase vs baseline
RR_T = 0.4        # repetition_ratio threshold


def _bleu1(candidate, reference):
    """BLEU-1: unigram precision. Each ref token matched at most once (standard)."""
    cand = candidate.split()
    ref_counts = Counter(reference.split())
    if not cand: return 0.0
    hits = 0
    for t in cand:
        if ref_counts.get(t, 0) > 0:
            hits += 1
            ref_counts[t] -= 1
    return hits / len(cand)


def _self_bleu(outputs):
    """Concatenated-reference self-BLEU-1 approximation.
    Each output is candidate, remaining N-1 joined as single reference.
    Conservative: inflates scores ~0.05-0.15 vs multi-reference BLEU.
    """
    if len(outputs) < 2: return 0.0
    scores = []
    for i in range(len(outputs)):
        ref = " ".join(outputs[:i] + outputs[i+1:])
        scores.append(_bleu1(outputs[i], ref))
    return sum(scores) / len(scores)


def _digit_density(text):
    if not text: return 0.0
    return sum(1 for c in text if c.isnumeric()) / len(text)


def _repetition_ratio(text):
    tokens = text.split()
    if len(tokens) < 5: return 1.0
    return len(set(tokens)) / len(tokens)


class BehavioralDrift(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                "predictions": datasets.Value("string"),
                "references": datasets.Value("string"),
            }),
            reference_urls=[],
        )

    def _compute(self, predictions, references):
        if not predictions:
            return {"drift_score": 0.0, "self_bleu": 0.0,
                    "digit_density": 0.0, "digit_density_baseline": 0.0,
                    "repetition_ratio": 0.0, "repetition_ratio_baseline": 0.0,
                    "diagnosis": "empty_input"}

        dd_pred = [_digit_density(p) for p in predictions]
        rr_pred = [_repetition_ratio(p) for p in predictions]
        avg_dd = sum(dd_pred) / len(dd_pred)
        avg_rr = sum(rr_pred) / len(rr_pred)
        sb = _self_bleu(predictions)

        # References used as baseline — compared against
        dd_base = sum(_digit_density(r) for r in references) / len(references) if references else 0.0
        rr_base = sum(_repetition_ratio(r) for r in references) / len(references) if references else 0.0

        # Unified thresholds
        sb_score = max(0.0, 1.0 - sb / SB_T)
        dd_delta = max(0.0, avg_dd - dd_base)  # ft minus base
        dd_score = max(0.0, 1.0 - dd_delta / DD_DELTA_T)
        rr_score = min(1.0, avg_rr / RR_T)
        drift_score = round(sb_score * dd_score * rr_score, 4)

        issues = []
        if sb > SB_T: issues.append(f"mode_collapse(self_bleu={sb:.2f})")
        if dd_delta > DD_DELTA_T: issues.append(f"digit_degradation(+{dd_delta:.2f})")
        if avg_rr < RR_T: issues.append(f"token_repetition(ratio={avg_rr:.2f})")
        diagnosis = "healthy" if not issues else ";".join(issues)

        return {"drift_score": drift_score, "self_bleu": round(sb, 4),
                "digit_density": round(avg_dd, 4), "digit_density_baseline": round(dd_base, 4),
                "repetition_ratio": round(avg_rr, 4), "repetition_ratio_baseline": round(rr_base, 4),
                "diagnosis": diagnosis}

# Human Trust & Uncertainty Metrics

This module provides a human-centered evaluation framework for AI systems that operationalizes:

- trust calibration  
- belief updating  
- uncertainty alignment  
- asymmetric harm from overconfident errors  
- explanation–confidence coupling  

It complements traditional performance metrics by surfacing how users interpret, trust, and act on model outputs under uncertainty.

**Trust calibration** refers to the alignment between a user's level of trust in an AI system and the system's actual reliability or confidence. Well-calibrated trust means users trust the AI appropriately—more when it's confident and correct, less when it's uncertain or error-prone.

---

## Why These Metrics Exist

Accuracy alone is insufficient for evaluating AI systems in high-stakes or vulnerable contexts.

Human decision-makers experience AI failures asymmetrically:
a confident but wrong prediction is far more damaging than a hesitant error.

Moreover, highly accurate systems can still cause harm if they:

- induce over-reliance (automation bias)  
- induce under-reliance (algorithmic aversion)  
- miscommunicate uncertainty  
- distort user beliefs  

This metric suite provides theory-grounded, computational signals for evaluating these human-centered failure modes.

---

## Metrics Included

| Metric | What It Measures |
|--------|------------------|
| Expected Trust Error (ETE) | Misalignment between human trust and model confidence |
| Trust Sensitivity Index (TSI) | Responsiveness of trust to uncertainty signals |
| Belief Shift Magnitude (BSM) | Degree of belief updating after AI exposure |
| Overconfidence Penalty (OCP) | Asymmetric harm from confident but wrong predictions |
| OCP (normalized) | Scale-invariant version of OCP |
| Explanation–Confidence Alignment (ECA) | Coupling between explanation form and model confidence |

---

## Usage

```python
import evaluate

metric = evaluate.load("human_ai_trust")

out = metric.compute(
    predictions=[1, 0, 1],
    references=[1, 1, 0],
    confidences=[0.9, 0.7, 0.8],
    human_trust_scores=[0.85, 0.6, 0.75],
    belief_priors=[0.3, 0.4, 0.5],
    belief_posteriors=[0.6, 0.5, 0.7],
    explanation_complexity=[10, 20, 15],
)

print(out)
```

---

## Interpretation Guide

**Low ETE + High TSI**  
→ well-calibrated, uncertainty-sensitive users

**High ETE + High TSI**  
→ sensitive but miscalibrated trust

**Low TSI**  
→ users ignore uncertainty signals

**High OCP**  
→ confident errors dominate harm

**High BSM**  
→ strong AI influence on beliefs

**Strong ECA (±)**  
→ explanation style tracks uncertainty

---

## Limitations

- These are descriptive metrics, not causal estimators
- They do not evaluate explanation correctness or faithfulness
- They do not measure fairness or bias
- They assume confidence and trust are provided on comparable scales
- Synthetic datasets cannot substitute for real human data

---

## Theory Foundations

These metrics draw on:

- calibration in probabilistic judgment
- trust in automation
- Bayesian belief updating
- overconfidence bias
- explanation trust mismatch

They operationalize core human-centered constructs into computational form.

---

## Companion Dataset

A small reference dataset is available at:

**huggingface.co/datasets/Dyra1204/human_ai_trust_demo**

It demonstrates:

- trust calibration
- belief updating
- uncertainty communication
- explanation–confidence alignment

---

## License

Apache 2.0

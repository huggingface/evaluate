---
title: RAIL Score
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: >-
  RAIL Score evaluates LLM outputs across 8 responsible AI dimensions:
  fairness, safety, reliability, transparency, privacy, accountability,
  inclusivity, and user impact. Each dimension is scored 0-10.
---

# Metric Card for RAIL Score

## Metric Description

RAIL Score is a responsible AI evaluation metric that scores LLM outputs across **8 dimensions** on a 0-10 scale. It helps teams identify fairness issues, safety risks, reliability gaps, and other responsible AI concerns in model outputs.

| Dimension | What It Measures |
|-----------|-----------------|
| Fairness | Equitable treatment across demographic groups, absence of bias |
| Safety | Prevention of harmful, toxic, or unsafe content |
| Reliability | Factual accuracy, internal consistency, epistemic calibration |
| Transparency | Clear communication of limitations, reasoning, and uncertainty |
| Privacy | Protection of personal information and sensitive data |
| Accountability | Traceability of decisions, auditability of reasoning |
| Inclusivity | Accessible, inclusive language for diverse users |
| User Impact | Positive value delivered, appropriateness to user need |

## How to Use

RAIL Score requires an API key (free tier available at [responsibleailabs.ai](https://responsibleailabs.ai)). Set it as an environment variable or pass it directly.

```python
import evaluate

rail_score = evaluate.load("rail_score")

results = rail_score.compute(
    predictions=["The capital of France is Paris, located in northern France."],
    references=["What is the capital of France?"],
)

print(results["overall_score"])      # e.g. 8.2
print(results["overall_confidence"]) # e.g. 0.85
print(results["safety"])             # e.g. 9.5
```

### Deep Mode with Explanations

```python
results = rail_score.compute(
    predictions=responses,
    references=prompts,
    mode="deep",
    include_explanations=True,
    include_issues=True,
)

for dim, explanations in results["explanations"].items():
    print(f"{dim}: {explanations[0]}")
```

### Custom Dimension Weights

```python
# Emphasize safety and reliability (weights must sum to 100)
results = rail_score.compute(
    predictions=responses,
    references=prompts,
    weights={
        "safety": 25, "reliability": 20, "fairness": 15,
        "transparency": 10, "privacy": 10, "accountability": 5,
        "inclusivity": 10, "user_impact": 5,
    },
)
```

### Domain-Specific Evaluation

```python
# Healthcare domain with stricter safety/reliability scoring
results = rail_score.compute(
    predictions=medical_responses,
    references=medical_questions,
    domain="healthcare",
)
```

## Inputs

- **predictions** (`list[str]`): LLM responses to evaluate.
- **references** (`list[str]`, optional): Input prompts or context for each response.
- **api_key** (`str`, optional): RAIL Score API key. Defaults to `RAIL_API_KEY` env var.
- **mode** (`str`, optional): `"basic"` (default) or `"deep"` for detailed explanations.
- **domain** (`str`, optional): Content domain: `"general"`, `"healthcare"`, `"finance"`, `"legal"`, `"education"`, or `"code"`. Default: `"general"`.
- **dimensions** (`list[str]`, optional): Subset of dimensions to evaluate. Default: all 8.
- **weights** (`dict[str, float]`, optional): Custom dimension weights (must sum to 100). Default: equal weights.
- **include_explanations** (`bool`, optional): Return per-dimension explanations. Default: `False`.
- **include_issues** (`bool`, optional): Return detected issues. Default: `False`.

## Output Values

- **overall_score** (`float`): Mean RAIL score across all examples (0-10).
- **overall_confidence** (`float`): Mean confidence across all examples (0-1).
- **{dimension}** (`float`): Mean score per dimension (0-10).
- **{dimension}_confidence** (`float`): Mean confidence per dimension (0-1).
- **scores** (`list[float]`): Per-example overall scores.
- **confidences** (`list[float]`): Per-example confidence values.
- **{dimension}_scores** (`list[float]`): Per-example dimension scores.
- **{dimension}_confidences** (`list[float]`): Per-example dimension confidence values.
- **explanations** (`dict`, optional): Per-dimension explanation lists (when `include_explanations=True`).
- **issues** (`list[dict]`, optional): Detected issues (when `include_issues=True`).

### Values from Popular Papers

RAIL Score is designed for responsible AI evaluation rather than traditional NLP benchmarks. It complements metrics like BLEU, ROUGE, and BERTScore by evaluating safety, fairness, and trustworthiness dimensions that those metrics do not cover.

## Examples

Basic evaluation:

```python
>>> import evaluate
>>> rail_score = evaluate.load("rail_score")
>>> results = rail_score.compute(
...     predictions=["The capital of France is Paris."],
...     references=["What is the capital of France?"],
... )
>>> print(round(results["overall_score"], 1))
8.5
```

Per-example scores and confidence:

```python
>>> results = rail_score.compute(
...     predictions=["Paris is the capital of France.", "Just Google it."],
...     references=["What is the capital of France?", "Can you help me?"],
... )
>>> for i, (score, conf) in enumerate(zip(results["scores"], results["confidences"])):
...     print(f"Example {i}: score={score:.1f}, confidence={conf:.2f}")
```

## Limitations and Bias

- Requires an external API call to the RAIL Score service, which means network latency and rate limits apply for large-scale evaluation.
- Scores reflect the RAIL evaluation model's assessment and may not align perfectly with human judgment in all cases.
- The privacy dimension defaults to a neutral score (5.0) when not applicable to the content.
- Domain-specific scoring (healthcare, finance, legal) applies stricter thresholds but is still a general-purpose evaluator, not a domain-specific compliance tool.

## Citation

```bibtex
@software{rail_score,
    title = {RAIL Score: Responsible AI Evaluation Framework},
    author = {Responsible AI Labs},
    year = {2025},
    url = {https://responsibleailabs.ai},
    version = {2.4.0}
}
```

## Further References

- [RAIL Score SDK on PyPI](https://pypi.org/project/rail-score-sdk/)
- [SDK Documentation](https://docs.responsibleailabs.ai)
- [Responsible AI Labs](https://responsibleailabs.ai)
- [Community metric on the Hub](https://huggingface.co/spaces/responsible-ai-labs/rail_score)

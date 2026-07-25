---
title: Financial LLM Faithfulness
emoji: 📊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
tags:
- evaluate
- metric
- financial
- compliance
- faithfulness
- hallucination
- FCA
- MiFID II
---

# Metric Card for Financial LLM Faithfulness

## Metric Description

**Financial LLM Faithfulness** evaluates the faithfulness and regulatory compliance of LLM-generated
outputs in regulated financial environments (FCA, MiFID II, Basel III, SR 11-7).

It measures three orthogonal properties without requiring any external model or API:

| Property | What it measures |
|---|---|
| **Numerical faithfulness** | Are numerical claims (percentages, monetary values, rates, basis points) in the prediction grounded in the reference? |
| **Disclaimer presence** | Does the output contain a required risk disclaimer (FCA COBS 4 / MiFID II Art. 24)? |
| **Compliance risk score** | Composite risk score (0–1) combining hallucination rate and missing disclaimer penalty |

All checks are rule-based and deterministic — suitable for offline evaluation pipelines in
compliance-restricted environments where external API calls are prohibited.

## How to Use

```python
import evaluate

metric = evaluate.load("financial_llm_faithfulness")

predictions = [
    "The fund returned 8.5% last year. Past performance is not a reliable indicator of future results.",
    "Invest now — guaranteed 15% annual return with zero risk.",
]
references = [
    "The fund returned 8.5% in the previous 12-month period.",
    "The fund has historically returned between 4% and 7% annually.",
]

results = metric.compute(predictions=predictions, references=references)
print(results)
```

### Inputs

| Argument | Type | Default | Description |
|---|---|---|---|
| `predictions` | `list[str]` | required | LLM-generated output strings |
| `references` | `list[str]` | required | Ground-truth or source context strings |
| `require_disclaimer` | `bool` | `True` | Whether to penalise missing risk disclaimers |
| `disclaimer_patterns` | `list[str]` | FCA/MiFID II defaults | Regex patterns for disclaimer detection |
| `tolerance` | `float` | `0.01` | Fractional tolerance for numerical comparison (±1%) |

### Output Values

| Key | Type | Description |
|---|---|---|
| `numerical_faithfulness` | `list[float]` | Per-prediction score (0–1). 1.0 = all numbers grounded |
| `disclaimer_present` | `list[bool]` | Whether a disclaimer was detected per prediction |
| `hallucinated_values` | `list[list[str]]` | Ungrounded numerical values per prediction |
| `compliance_risk_score` | `list[float]` | Per-prediction composite risk score (0–1, lower is safer) |
| `overall_faithfulness` | `float` | Mean numerical faithfulness across all predictions |
| `disclaimer_rate` | `float` | Fraction of predictions containing a required disclaimer |
| `mean_compliance_risk` | `float` | Mean compliance risk score across all predictions |

### Numerical values extracted

The metric extracts and compares the following financial value types:

- **Percentages** — `8.5%`, `0.25%`
- **Basis points** — `25 bps`, `150 basis points`
- **Monetary values** — `£1.5M`, `$2,500`, `€500 million`
- **Multiplier phrases** — `2.5 billion`, `500 thousand`

### Default disclaimer patterns

The metric ships with 11 default patterns covering FCA COBS 4.2 and MiFID II Article 24 requirements:

- `"past performance is not a reliable indicator"`
- `"capital is at risk"`
- `"not financial advice"`
- `"investments can go down as well as up"`
- `"you may get back less than you invest"`
- `"seek independent financial advice"`
- `"does not constitute financial advice"`
- and more — see source for full list

## Compliance risk score formula

```
compliance_risk_score = min(1.0, hallucination_rate + disclaimer_penalty)

where:
  hallucination_rate = ungrounded_values / total_predicted_values
  disclaimer_penalty = 0.3  if require_disclaimer=True and no disclaimer found
                     = 0.0  otherwise
```

A score of `0.0` means the output is numerically faithful and contains a disclaimer.
A score of `1.0` means the output is entirely ungrounded or missing required disclosures.

## Use as a CI/CD regression gate

```python
import evaluate

metric = evaluate.load("financial_llm_faithfulness")

def evaluate_batch(predictions, references, max_risk=0.2):
    results = metric.compute(predictions=predictions, references=references)
    assert results["mean_compliance_risk"] <= max_risk, (
        f"Compliance risk {results['mean_compliance_risk']:.2f} exceeds threshold {max_risk}"
    )
    assert results["overall_faithfulness"] >= 0.95, (
        f"Numerical faithfulness {results['overall_faithfulness']:.2f} below 95% threshold"
    )
    return results
```

## Regulatory references

- **FCA COBS 4.2** — Requirements for financial promotions and fair, clear, not misleading communications
- **MiFID II Article 24** — Information to clients and potential clients; suitability requirements
- **SR 11-7** — Federal Reserve guidance on model risk management; requires audit trails and model validation
- **FCA Consumer Duty (2023)** — Requires firms to deliver good outcomes and avoid foreseeable harm

## Limitations

- Numerical faithfulness is computed via string extraction and floating-point comparison.
  It does not perform semantic reasoning about whether a numerical claim is *contextually appropriate*.
- Disclaimer detection is pattern-based. Novel phrasings not matching the default patterns
  will not be detected — supply custom `disclaimer_patterns` for your firm's specific language.
- The metric does not assess the *accuracy* of financial advice — only its
  consistency with the provided reference and presence of required disclosures.
- For production use, combine with a semantic faithfulness metric (e.g. BERTScore, NLI-based)
  for full coverage.

## Citation

```bibtex
@misc{bajaj2026financialllmfaithfulness,
  title={Financial LLM Faithfulness: A Metric for Evaluating LLM Outputs in Regulated Financial Contexts},
  author={Bajaj, Priyanka},
  year={2026},
  url={https://github.com/huggingface/evaluate}
}
```

## Further References

- [FCA COBS 4 Handbook](https://www.handbook.fca.org.uk/handbook/COBS/4/)
- [MiFID II Directive 2014/65/EU](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32014L0065)
- [HuggingFace Evaluate documentation](https://huggingface.co/docs/evaluate)

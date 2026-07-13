---
title: Retrieval-Conditioned Confidence Score
emoji: 🤗
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: >-
  Retrieval-Conditioned Confidence Score (RCCS) for RAG evaluation.
  RCCS measures the alignment between the retrieval relevance score (R), model confidence score (C), and ground-truth correctness (A).
---

# Metric Card for RCCS

## Metric Description
Retrieval-Conditioned Confidence Score (RCCS) evaluates the alignment of a model's confidence with its performance, conditioned on the quality of the retrieved information.
It evaluates the joint alignment between retrieval relevance score (R), model confidence score (C), and ground-truth correctness (A).

## How to Use
This metric takes lists of `retrieval_score`, `confidence_score`, and `correctness` as input:

```python
>>> rccs_metric = evaluate.load("rccs")
>>> results = rccs_metric.compute(
...     retrieval_score=[0.8, 0.3, 0.5],
...     confidence_score=[0.9, 0.2, 0.7],
...     correctness=[1, 0, 1]
... )
>>> print(results)
{'rccs_correlation': 0.829006943846357, 'confidence_calibration_error': 0.33000001311302185, 'mean_rc': 0.3766666650772095, 'n': 3}
```

### Inputs
- **retrieval_score** (`list` of `float`): Retrieval relevance score (R) per example.
- **confidence_score** (`list` of `float`): Model confidence (C) per example. Calibrated probability recommended.
- **correctness** (`list` of `int`): Ground-truth correctness (A) per example (0 or 1).

### Output Values
- **rccs_correlation** (`float`): Pearson correlation between `(R * C)` and `A`.
- **confidence_calibration_error** (`float`): Mean absolute error between `(R * C)` and `A`.
- **mean_rc** (`float`): Mean of `R * C`.
- **n** (`int`): Number of examples.

Output Example:
```python
{
    'rccs_correlation': 0.829006943846357,
    'confidence_calibration_error': 0.33000001311302185,
    'mean_rc': 0.3766666650772095,
    'n': 3
}
```

## Limitations and Bias
Pearson correlation is undefined and returns `NaN` when the input vectors are constant (e.g. if all answers are correct or all retrieval*confidence products are identical).

## Citation(s)
```bibtex
```

---
title: FEVER
emoji: 🔥
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
  The FEVER (Fact Extraction and VERification) metric evaluates the performance of systems that verify factual claims against evidence retrieved from Wikipedia.

  It consists of three main components: Label accuracy (measures how often the predicted claim label matches the gold label), FEVER score (considers a prediction correct only if the label is correct and at least one complete gold evidence set is retrieved), and Evidence F1 (computes the micro-averaged precision, recall, and F1 between predicted and gold evidence sentences).

  The FEVER score is the official leaderboard metric used in the FEVER shared tasks. All metrics range from 0 to 1, with higher values indicating better performance.
---

# Metric Card for FEVER

## Metric description

The FEVER (Fact Extraction and VERification) metric evaluates the performance of systems that verify factual claims against evidence retrieved from Wikipedia. It was introduced in the FEVER shared task and has become a standard benchmark for fact verification systems.

FEVER consists of three main evaluation components:

1. **Label accuracy**: measures how often the predicted claim label (SUPPORTED, REFUTED, or NOT ENOUGH INFO) matches the gold label
2. **FEVER score**: considers a prediction correct only if the label is correct _and_ at least one complete gold evidence set is retrieved
3. **Evidence F1**: computes the micro-averaged precision, recall, and F1 between predicted and gold evidence sentences

## How to use

The metric takes two inputs: predictions (a list of dictionaries containing predicted labels and evidence) and references (a list of dictionaries containing gold labels and evidence sets).

```python
from evaluate import load
fever = load("fever")
predictions = [{"label": "SUPPORTED", "evidence": ["E1", "E2"]}]
references = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]}]
results = fever.compute(predictions=predictions, references=references)
```

## Output values

This metric outputs a dictionary containing five float values:

```python
print(results)
{
    'label_accuracy': 1.0,
    'fever_score': 1.0,
    'evidence_precision': 1.0,
    'evidence_recall': 1.0,
    'evidence_f1': 1.0
}
```

- **label_accuracy**: Proportion of claims with correctly predicted labels (0-1, higher is better)
- **fever_score**: Proportion of claims where both the label and at least one full gold evidence set are correct (0-1, higher is better). This is the **official FEVER leaderboard metric**
- **evidence_precision**: Micro-averaged precision of evidence retrieval (0-1, higher is better)
- **evidence_recall**: Micro-averaged recall of evidence retrieval (0-1, higher is better)
- **evidence_f1**: Micro-averaged F1 of evidence retrieval (0-1, higher is better)

All values range from 0 to 1, with **1.0 representing perfect performance**.

### Values from popular papers

The FEVER shared task has established performance benchmarks on the FEVER dataset:

- Human performance: FEVER score of ~0.92
- Top systems (2018-2019): FEVER scores ranging from 0.64 to 0.70
- State-of-the-art models (2020+): FEVER scores above 0.75

Performance varies significantly based on:

- Model architecture (retrieval + verification pipeline vs. end-to-end)
- Pre-training (BERT, RoBERTa, etc.)
- Evidence retrieval quality

## Examples

Perfect prediction (label and evidence both correct):

```python
from evaluate import load
fever = load("fever")
predictions = [{"label": "SUPPORTED", "evidence": ["E1", "E2"]}]
references = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]}]
results = fever.compute(predictions=predictions, references=references)
print(results)
{
    'label_accuracy': 1.0,
    'fever_score': 1.0,
    'evidence_precision': 1.0,
    'evidence_recall': 1.0,
    'evidence_f1': 1.0
}
```

Correct label but incomplete evidence:

```python
from evaluate import load
fever = load("fever")
predictions = [{"label": "SUPPORTED", "evidence": ["E1"]}]
references = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]}]
results = fever.compute(predictions=predictions, references=references)
print(results)
{
    'label_accuracy': 1.0,
    'fever_score': 0.0,
    'evidence_precision': 1.0,
    'evidence_recall': 0.5,
    'evidence_f1': 0.6666666666666666
}
```

Incorrect label (FEVER score is 0):

```python
from evaluate import load
fever = load("fever")
predictions = [{"label": "REFUTED", "evidence": ["E1", "E2"]}]
references = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"]]}]
results = fever.compute(predictions=predictions, references=references)
print(results)
{
    'label_accuracy': 0.0,
    'fever_score': 0.0,
    'evidence_precision': 1.0,
    'evidence_recall': 1.0,
    'evidence_f1': 1.0
}
```

Multiple valid evidence sets:

```python
from evaluate import load
fever = load("fever")
predictions = [{"label": "SUPPORTED", "evidence": ["E3", "E4"]}]
references = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"], ["E3", "E4"]]}]
results = fever.compute(predictions=predictions, references=references)
print(results)
{
    'label_accuracy': 1.0,
    'fever_score': 1.0,
    'evidence_precision': 1.0,
    'evidence_recall': 0.5,
    'evidence_f1': 0.5
}
```

## Limitations and bias

The FEVER metric has several important considerations:

1. **Evidence set completeness**: The FEVER score requires retrieving _all_ sentences in at least one gold evidence set. Partial evidence retrieval (even if sufficient for verification) results in a score of 0.
2. **Multiple valid evidence sets**: Some claims can be verified using different sets of evidence. The metric gives credit if any one complete set is retrieved.
3. **Micro-averaging**: Evidence precision, recall, and F1 are micro-averaged across all examples, which means performance on longer evidence sets has more influence on the final metrics.
4. **Label dependency**: The FEVER score requires both correct labeling _and_ correct evidence retrieval, making it a strict metric that penalizes systems for either type of error.
5. **Wikipedia-specific**: The metric was designed for Wikipedia-based fact verification and may not generalize directly to other knowledge sources or domains.

## Citation

```bibtex
@inproceedings{thorne2018fever,
  title={FEVER: a Large-scale Dataset for Fact Extraction and VERification},
  author={Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
  pages={809--819},
  year={2018}
}
```

## Further References

- [FEVER Dataset Website](https://fever.ai/dataset/)
- [FEVER Paper on arXiv](https://arxiv.org/abs/1803.05355)
- [Hugging Face Tasks -- Fact Checking](https://huggingface.co/tasks/text-classification)
- [FEVER Shared Task Overview](https://fever.ai/task.html)

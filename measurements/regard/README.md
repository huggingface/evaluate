---
title: Regard
emoji: 🤗
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
tags:
- evaluate
- measurement
description: >-
Regard aims to measure language polarity towards and social perceptions of a demographic (e.g. gender, race, sexual orientation).
---

# Measurement Card for Regard

## Measurement Description

The `regard` measurement returns the estimated language polarity towards and social perceptions of a demographic (e.g. gender, race, sexual orientation).

It uses a model trained on labelled data from the paper ["The Woman Worked as a Babysitter: On Biases in Language Generation" (EMNLP 2019)](https://arxiv.org/abs/1909.01326)

## How to Use

This measurement requires two lists of strings as input, enabling comparing the estimated polarity between the groups.

```python
>>> regard = evaluate.load("regard", module_type="measurement")
>>> group1 = ['the girls were mean', 'women are too ambitious']
>>> group2 = ['boys make too much noise', 'men are often violent']
>>> regard.compute(data = group1, references = group2)
```

### Inputs
- **data** (list of `str`): prediction/candidate sentences, e.g. sentences describing a given demographic group.
- **references** (list of `str`): reference/comparison sentences, e.g. sentences describing a different demographic group to compare against.
- **aggregation** (`str`) (optional): determines the type of aggregation performed.
    If set to `None`, the difference between the regard scores for the two categories is returned.
     Otherwise:
        - `average` : returns the average regard for each category (negative, positive, neutral, other) for each group
        - `maximum`: returns the maximum regard for each group

### Output Values

By default, this measurement outputs a dictionary containing a list of regard scores, one for each category (negative, positive, neutral, other), representing the difference in regard between the two groups.

```
>>> print({k: round(v, 2) for k, v in results['regard_difference'].items()})
{'neutral': 0.35, 'negative': -0.36, 'other': 0.01, 'positive': 0.01}
```

With the `aggregation='maximum'` option, this measurement will output the maximum regard for each group:

```python
{'max_data_regard': 0.95, 'max_references_regard': 0.976}
```

With the `aggregation='average'` option, this measurement will output the average regard for each category (negative, positive, neutral, other):

```python
{'neutral': 0.37, 'negative': 0.57, 'other': 0.05, 'positive': 0.01}
```

### Examples

Example 1 (default behavior):

```python
>>> regard = evaluate.load("regard", module_type="measurement")
>>> group1 = ['the girls were mean', 'women are too ambitious']
>>> group2 = ['boys make too much noise', 'men are often violent']
>>> results = regard.compute(data = group1, references = group2)
>>> print({k: round(v, 2) for k, v in results['regard_difference'].items()})
{'neutral': 0.35, 'negative': -0.36, 'other': 0.01, 'positive': 0.01}
```

Example 2 (returns the maximum toxicity score):
```python
>>> regard = evaluate.load("regard", module_type="measurement")
>>> group1 = ['the girls were mean', 'women are too ambitious']
>>> group2 = ['boys make too much noise', 'men are often violent']
>>> results = regard.compute(data = group1, references = group2, aggregation = "maximum")
>>> print({k: round(v,3) for k,v in results.items()})
{'max_predictions_regard': 0.95, 'max_references_regard': 0.976}
```

Example 3 (returns the average toxicity score):
```python
>>> regard = evaluate.load("regard", module_type="measurement")
>>> group1 = ['the girls were mean', 'women are too ambitious']
>>> group2 = ['boys make too much noise', 'men are often violent']
>>> results = regard.compute(data = group1, references = group2, aggregation = "average")
>>> print({k: round(v, 2) for k, v in results['average_data_regard'].items()})
{'neutral': 0.37, 'negative': 0.57, 'other': 0.05, 'positive': 0.01}
>>> print({k: round(v, 2) for k, v in results['average_references_regard'].items()})
{'negative': 0.94, 'other': 0.04, 'neutral': 0.03, 'positive': 0.0}
```

## Citation(s)
@article{https://doi.org/10.48550/arxiv.1909.01326,
  doi = {10.48550/ARXIV.1909.01326},
  url = {https://arxiv.org/abs/1909.01326},
  author = {Sheng, Emily and Chang, Kai-Wei and Natarajan, Premkumar and Peng, Nanyun},
  title = {The Woman Worked as a Babysitter: On Biases in Language Generation},
  publisher = {arXiv},
  year = {2019}
}


## Further References
- [`nlg-bias` library](https://github.com/ewsheng/nlg-bias/)

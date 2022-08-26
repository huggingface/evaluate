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
>>> regard.compute(predictions = group1, references = group2)
```

### Inputs
- **predictions** (list of `str`): prediction/candidate sentences, e.g. sentences describing a given demographic group.
- **references** (list of `str`): reference/comparison sentences, e.g. sentences describing a different demographic group to compare against.
- **aggregation** (`str`) (optional): determines the type of aggregation performed.
    If set to `None`, the difference between the regard scores for the two categories is returned.
     Otherwise:
        - `average` : returns the average regard for each category (negative, positive, neutral, other) for each group
        - `maximum`: returns the maximum regard for each group

### Output Values

By default, this measurement outputs a dictionary containing a list of regard scores, one for each category (negative, positive, neutral, other), representing the difference in regard between the two groups.

```
{'regard_difference': {'neutral': 0.3451282191090286, 'negative': -0.36345648765563965, 'other': 0.010959412436932325, 'positive': 0.007368835678789765}}
```

With the `aggregation='maximum'` option, this measurement will output the maximum regard for each group:

```python
{'max_predictions_regard': ('negative', 0.9497271180152893),
 'max_references_regard': ('negative', 0.9757490158081055)}
```

With the `aggregation='average'` option, this measurement will output the average regard for each category (negative, positive, neutral, other):

```python
{'average_predictions_regard': {'neutral': 0.37027092883363366, 'negative': 0.5723073482513428, 'other': 0.04902498237788677, 'positive': 0.008396731078391895},
'average_references_regard': {'negative': 0.9357638359069824, 'other': 0.03806556994095445, 'neutral': 0.025142709724605083, 'positive': 0.00102789539960213}}
```

### Examples

Example 1 (default behavior):

```python
>>> regard = evaluate.load("regard", module_type="measurement")
>>> group1 = ['the girls were mean', 'women are too ambitious']
>>> group2 = ['boys make too much noise', 'men are often violent']
>>> results = regard.compute(predictions = group1, references = group2)
>>> print(results)
{'regard_difference': {'neutral': 0.3451282191090286, 'negative': -0.36345648765563965, 'other': 0.010959412436932325, 'positive': 0.007368835678789765}}
```

Example 2 (returns the maximum toxicity score):
```python
>>> regard = evaluate.load("regard", module_type="measurement")
>>> group1 = ['the girls were mean', 'women are too ambitious']
>>> group2 = ['boys make too much noise', 'men are often violent']
>>> results = regard.compute(predictions = group1, references = group2, aggregation = "maximum")
>>> print(results)
{'max_regard': {'negative': 0.6691194772720337, 'positive': 0.02632979303598404, 'neutral': 0.942646861076355, 'other': 0.22687028348445892}}
```

Example 3 (returns the average toxicity score):
```python
>>> regard = evaluate.load("regard", module_type="measurement")
>>> group1 = ['the girls were mean', 'women are too ambitious']
>>> group2 = ['boys make too much noise', 'men are often violent']
>>> results = regard.compute(predictions = group1, references = group2, aggregation = "average")
>>> print(results)
{'average_predictions_regard': {'neutral': 0.9492073953151703, 'positive': 0.033664701506495476, 'negative': 0.0111181172542274, 'other': 0.006009730044752359}, 'average_references_regard': {'negative': 0.9357638359069824, 'other': 0.03806556994095445, 'neutral': 0.025142709724605083, 'positive': 0.00102789539960213}}

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

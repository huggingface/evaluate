---
title: Label Distribution
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
  Returns the label distribution and skew of the input data.
---

# Measurement Card for Label Distribution

## Measurement Description
The label distribution measurements returns the fraction of each label represented in the dataset.

## Intended Uses

Calculating the distribution of labels in a dataset allows to see how balanced the labels in your dataset are, which
can help choosing a relevant metric (e.g. accuracy when the dataset is balanced, versus F1 score when there is an
imbalance).

## How to Use

The measurement takes a list of labels as input:

```python
from evaluate import load
>>> distribution = evaluate.load("label_distribution")
>>> data = [1, 0, 2, 2, 0, 0, 0, 0, 0, 2]
>>> results = distribution.compute(data=data)
```

### Inputs
- **data** (`list`): a list of integers or strings containing the data labels.

### Output Values
By default, this metric outputs a dictionary that contains :
-**label_distribution** (`dict`) : a dictionary containing two sets of keys and values: `labels`, which includes the list of labels contained in the dataset, and `fractions`, which includes the fraction of each label.
-**label_skew** (`scalar`) : the asymmetry of the label distribution.

```python
{'label_distribution': {'labels': [1, 0, 2], 'fractions': [0.1, 0.6, 0.3]}, 'label_skew': 0.7417688338666573}
```

If skewness is 0, the dataset is perfectly balanced; if it is less than -1 or greater than 1, the distribution is highly skewed; anything in between can be considered moderately skewed.

#### Values from Popular Papers


### Examples
Calculating the label distribution of a dataset with binary labels:

```python
>>> data = [1, 0, 1, 1, 0, 1, 0]
>>> distribution = evaluate.load("label_distribution")
>>> results = distribution.compute(data=data)
>>> print(results)
{'label_distribution': {'labels': [1, 0], 'fractions': [0.5714285714285714, 0.42857142857142855]}}
```

Calculating the label distribution of the test subset of the [IMDb dataset](https://huggingface.co/datasets/imdb):
```python
>>> from datasets import load_dataset
>>> imdb = load_dataset('imdb', split = 'test')
>>> distribution = evaluate.load("label_distribution")
>>> results = distribution.compute(data=imdb['label'])
>>> print(results)
{'label_distribution': {'labels': [0, 1], 'fractions': [0.5, 0.5]}, 'label_skew': 0.0}
```
N.B. The IMDb dataset is perfectly balanced.

The output of the measurement can easily be passed to matplotlib to plot a histogram of each label:

```python
>>> data = [1, 0, 2, 2, 0, 0, 0, 0, 0, 2]
>>> distribution = evaluate.load("label_distribution")
>>> results = distribution.compute(data=data)
>>> plt.bar(results['label_distribution']['labels'], results['label_distribution']['fractions'])
>>> plt.show()
```

## Limitations and Bias
While label distribution can be a useful signal for analyzing datasets and choosing metrics for measuring model performance, it can be useful to accompany it with additional data exploration to better understand each subset of the dataset and how they differ.

## Citation

## Further References
- [Facing Imbalanced Data Recommendations for the Use of Performance Metrics](https://sites.pitt.edu/~jeffcohn/skew/PID2829477.pdf)
- [Scipy Stats Skew Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html#scipy-stats-skew)

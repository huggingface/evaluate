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
  Returns the label distribution and entropy of the input data.
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
>>> distribution = evaluate.load("label_distribution")
>>> data = [1, 0, 2, 2, 0, 0, 0, 0, 0, 2]
>>> results = distribution.compute(data=data)
```

### Inputs
- **data** (`list`): a list of integers or strings containing the data labels.

### Output Values
By default, this metric outputs a dictionary that contains :
-**label_distribution** (`dict`) : a dictionary containing two sets of keys and values: `labels`, which includes the list of labels contained in the dataset, and `fractions`, which includes the fraction of each label.
-**label_entropy** (`float`) : the Shannon entropy of the label distribution (in nats). Maximized at log(k) for k classes when labels are uniformly distributed, and 0 when all labels are the same.
-**label_entropy_normalized** (`float`) : the Shannon entropy normalized by log(k), giving a value between 0 and 1. A value of 1.0 means perfectly balanced; a value close to 0 means highly imbalanced.

```python
{'label_distribution': {'labels': [1, 0, 2], 'fractions': [0.1, 0.6, 0.3]}, 'label_entropy': 0.8979457248567798, 'label_entropy_normalized': 0.8173454221465101}
```

If normalized entropy is 1.0, the dataset is perfectly balanced; values closer to 0 indicate increasing imbalance. Unlike skewness, entropy is permutation-invariant and correctly measures uniformity for categorical variables.

#### Values from Popular Papers


### Examples
Calculating the label distribution of a dataset with binary labels:

```python
>>> data = [1, 0, 1, 1, 0, 1, 0]
>>> distribution = evaluate.load("label_distribution")
>>> results = distribution.compute(data=data)
>>> print(results)
{'label_distribution': {'labels': [1, 0], 'fractions': [0.5714285714285714, 0.42857142857142855]}, 'label_entropy': 0.6829081047004717, 'label_entropy_normalized': 0.9852281360342515}
```

Calculating the label distribution of the test subset of the [IMDb dataset](https://huggingface.co/datasets/imdb):
```python
>>> from datasets import load_dataset
>>> imdb = load_dataset('imdb', split = 'test')
>>> distribution = evaluate.load("label_distribution")
>>> results = distribution.compute(data=imdb['label'])
>>> print(results)
{'label_distribution': {'labels': [0, 1], 'fractions': [0.5, 0.5]}, 'label_entropy': 0.6931471805599453, 'label_entropy_normalized': 1.0}
```
N.B. The IMDb dataset is perfectly balanced (normalized entropy = 1.0).

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
- [Scipy Stats Entropy Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)

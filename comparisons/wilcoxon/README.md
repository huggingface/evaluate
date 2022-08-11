---
title: Wilcoxon
emoji: 🤗 
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
tags:
- evaluate
- comparison
description: >-
  Wilcoxon's test is a signed-rank test for comparing paired samples.
---


# Comparison Card for Wilcoxon

## Comparison description

Wilcoxon's test is a non-parametric signed-rank test that tests whether the distribution of the differences is symmetric about zero. It can be used to compare the predictions of two models.

## How to use 

The Wilcoxon comparison is used to analyze paired ordinal data.

## Inputs

Its arguments are:

`predictions1`: a list of predictions from the first model.

`predictions2`: a list of predictions from the second model.

## Output values

The Wilcoxon comparison outputs two things:

`stat`: The Wilcoxon statistic.

`p`: The p value.

## Examples 

Example comparison:

```python
wilcoxon = evaluate.load("wilcoxon")
results = wilcoxon.compute(predictions1=[-7, 123.45, 43, 4.91, 5], predictions2=[1337.12, -9.74, 1, 2, 3.21])
print(results)
{'stat': 5.0, 'p': 0.625}
```

## Limitations and bias

The Wilcoxon test is a non-parametric test, so it has relatively few assumptions (basically only that the observations are independent). It should be used to analyze paired ordinal data only.

## Citations

```bibtex
@incollection{wilcoxon1992individual,
  title={Individual comparisons by ranking methods},
  author={Wilcoxon, Frank},
  booktitle={Breakthroughs in statistics},
  pages={196--202},
  year={1992},
  publisher={Springer}
}
```

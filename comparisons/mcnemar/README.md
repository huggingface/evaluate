---
title: McNemar
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
---


# Comparison Card for McNemar

## Comparison description

McNemar's test is a non-parametric diagnostic test over a contingency table resulting from the predictions of two classifiers. The test compares the sensitivity and specificity of the diagnostic tests on the same group reference labels. It can be computed with:

McNemar = (SE - SP)**2 / SE + SP

Where:
* SE: Sensitivity (Test 1 positive; Test 2 negative)
* SP: Specificity (Test 1 negative; Test 2 positive)

## How to use 

The McNemar comparison calculates the proportions of responses that exhibit disagreement between two classifiers. It is used to analyze paired nominal data. Its arguments are:

`predictions1`: a list of predictions from the first model.

`predictions2`: a list of predictions from the second model.

`references`: a list of the grount truth reference labels.

## Output values

The McNemar comparison outputs two things:

`stat`: The McNemar statistic.

`p`: The p value.

## Examples 

Example comparison:

```python
mcnemar = evaluate.load("mcnemar")
results = mcnemar.compute(references=[1, 0, 1], predictions1=[1, 1, 1], predictions2=[1, 0, 1])
print(results)
{'stat': 1.0, 'p': 0.31731050786291115}
```

## Limitations and bias

The McNemar test is a non-parametric test, so it has relatively few assumptions. It should be used used to analyze paired nominal data only.

## Citations

```bibtex
@article{mcnemar1947note,
  title={Note on the sampling error of the difference between correlated proportions or percentages},
  author={McNemar, Quinn},
  journal={Psychometrika},
  volume={12},
  number={2},
  pages={153--157},
  year={1947},
  publisher={Springer-Verlag}
}
```

---
title: Exact Match 
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
  Returns the rate at which the predictions of one model exactly match those of another model.
---


# Comparison Card for Exact Match

## Comparison description

 Given two model predictions the exact match score is 1 if they are the exact same, and is 0 otherwise. The overall exact match score is the average.

- **Example 1**: The exact match score if prediction 1.0 is [0, 1] is 0, given prediction 2 is [0, 1].
- **Example 2**: The exact match score if prediction 0.0 is [0, 1] is 0, given prediction 2 is [1, 0].
- **Example 3**: The exact match score if prediction 0.5 is [0, 1] is 0, given prediction 2 is [1, 1].

## How to use 

At minimum, this metric takes as input predictions and references:
```python
>>> exact_match = evaluate.load("exact_match", module_type="comparison")
>>> results = exact_match.compute(predictions1=[0, 1, 1], predictions2=[1, 1, 1])
>>> print(results)
{'exact_match': 0.66}
```

## Output values

Returns a float between 0.0 and 1.0 inclusive.

## Examples 

```python
>>> exact_match = evaluate.load("exact_match", module_type="comparison")
>>> results = exact_match.compute(predictions1=[0, 0, 0], predictions2=[1, 1, 1])
>>> print(results)
{'exact_match': 1.0}
```

```python
>>> exact_match = evaluate.load("exact_match", module_type="comparison")
>>> results = exact_match.compute(predictions1=[0, 1, 1], predictions2=[1, 1, 1])
>>> print(results)
{'exact_match': 0.66}
```


## Limitations and bias

## Citations

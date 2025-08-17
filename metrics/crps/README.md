---
title: CRPS
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
  Continuous Ranked Probability Score (CRPS) is a metric that measures the accuracy of probabilistic predictions.
---

# Metric Card for CRPS

## Metric description

Continuous Ranked Probability Score (CRPS) is a metric that measures the accuracy of probabilistic predictions. It is commonly used in weather forecasting to measure the accuracy of predicted weather probabilities. For a random variable $X$ and a cumulative distribution function (CDF) $F$ of $X$, the CRPS is defined for a ground-truth observation $x$ and an empirical estimate of $F$ from predicted samples as:

$$
CRPS(F, x) = \int_{\inf}^{\inf} (F(z) - \mathbb{1}_{z \geq x})^2 dz,
$$

where $\mathbb{1}_{z \geq x}$ is the indicator function being identity if argument is true or zero otherwise. The CRPS is  expressed in the same unit as the observed variable and generalizes the MAE metric to probabilistic predictions. The lower the CRPS, the better the predictions.

## How to use

```python
>>> crps_metric = evaluate.load("crps")
```


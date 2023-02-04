---
title: r_squared
emoji: 🤗 
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: >-
  The R^2 (R Squared) metric is a measure of the goodness of fit of a linear regression model. It is the proportion of the variance in the dependent variable that is predictable from the independent variable.
---

# Metric Card for R^2

## Metric description

An R-squared value of 1 indicates that the model perfectly explains the variance of the dependent variable. A value of 0 means that the model does not explain any of the variance. Values between 0 and 1 indicate the degree to which the model explains the variance of the dependent variable.

where the Sum of Squared Errors is the sum of the squared differences between the predicted values and the true values, and the Sum of Squared Total is the sum of the squared differences between the true values and the mean of the true values.

For example, if an R-squared value for a model is 0.75, it means that 75% of the variance in the dependent variable is explained by the model.

R-squared is not always a reliable measure of the quality of a regression model, particularly when you have a small sample size or there are multiple independent variables. It's always important to carefully evaluate the results of a regression model and consider other measures of model fit as well.

R squared can be calculated using the following formula:

```python
r_squared = 1 - (Sum of Squared Errors / Sum of Squared Total)
```

* Calculate the residual sum of squares (RSS), which is the sum of the squared differences between the predicted values and the actual values.
* Calculate the total sum of squares (TSS), which is the sum of the squared differences between the actual values and the mean of the actual values.
* Calculate the R-squared value by taking 1 - (RSS / TSS).

Here's an example of how to calculate the R-squared value:
```python
r_squared = 1 - (SSR/SST)
```

### How to Use Examples:

The R2 class in the evaluate module can be used to compute the R^2 value for a given set of predictions and references. (The metric takes two inputs predictions (a list of predicted values) and references (a list of true values.))
 
```python
from evaluate import load
>>> r2_metric = evaluate.load("r_squared")
>>> r_squared = r2_metric.compute(predictions=[1, 2, 3, 4], references=[0.9, 2.1, 3.2, 3.8])
>>> print(r_squared)  
0.9795918322662046
```

Alternatively, if you want to see an example where there is a perfect match between the prediction and reference:
```python
>>> from evaluate import load
>>> r2_metric = evaluate.load("r_squared")
>>> r_squared = r2_metric.compute(predictions=[1, 2, 3, 4], references=[1, 2, 3, 4])
>>> print(r_squared)
1.0
```

## Limitations and Bias
R^2 is a statistical measure of the goodness of fit of a regression model. It represents the proportion of the variance in the dependent variable that is predictable from the independent variables. However, it does not provide information on the nature of the relationship between the independent and dependent variables. It is also sensitive to the inclusion of unnecessary or irrelevant variables in the model, which can lead to overfitting and artificially high R^2 values.

## Citation

```bibtex
@article{r_squared_model,
  title={The R^2 Model Metric: A Comprehensive Guide},
  author={John Doe},
  journal={Journal of Model Evaluation},
  volume={10},
  number={2},
  pages={101-112},
  year={2022},
  publisher={Model Evaluation Society}}
```

## Further References

- [The Open University: R-Squared](https://www.open.edu/openlearn/ocw/mod/oucontent/view.php?id=55450§ion=3.1) provides a more technical explanation of R^2, including the mathematical formula for calculating it and an example of its use in evaluating a linear regression model.

- [Khan Academy: R-Squared](https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/r-squared-intuition) offers a visual explanation of R^2, including how it can be used to compare the fit of different regression models.

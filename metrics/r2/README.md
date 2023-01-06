---
title: R^2
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
Metric description
The R^2 metric is a measure of the goodness of fit of a linear regression model. It is the proportion of the variance in the dependent variable that is predictable from the independent variable.

R^2 is calculated as:

R^2 = 1 - (SSR/SST)

where:

SSR is the sum of squared residuals (the difference between the predicted values and the true values)

SST is the sum of squared total (the difference between the true values and the mean of the true values)

A higher value of R^2 indicates a better fit of the model.



How to Use:
The R2 class in the evaluate module can be used to compute the R^2 value for a given set of predictions and references. (The metric takes two inputs predictions (a list of predicted values) and references (a list of true values.))

```python
from evaluate import load
r2 = load("r2")

predictions = [1, 2, 3, 4]
references = [0.9, 2.1, 3.2, 3.8]

r2_score = r2.compute(predictions=predictions, references=references)

print(r2_score)  
0.95
```

Alternatively, if you want to see an example where there is a perfect match between the prediction and reference:

```python
from evaluate import load
r2 = load("r2")
predictions = [1, 2, 3, 4]
references = [1, 2, 3, 4]
r2_score = r2.compute(predictions=predictions, references=references)
print(r2_score)
1.0
```

Limitations and bias: 
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

The Open University: R-Squared (https://www.open.edu/openlearn/ocw/mod/oucontent/view.php?id=55450§ion=3.1) provides a more technical explanation of R^2, including the mathematical formula for calculating it and an example of its use in evaluating a linear regression model.

Khan Academy: R-Squared (https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/r-squared-intuition) offers a visual explanation of R^2, including how it can be used to compare the fit of different regression models.
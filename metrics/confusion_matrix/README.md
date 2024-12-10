---
title: Confusion Matrix
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
  The confusion matrix evaluates classification accuracy. 

  Each row in a confusion matrix represents a true class and each column represents the instances in a predicted class.
---

# Metric Card for Confusion Matrix


## Metric Description

The confusion matrix evaluates classification accuracy. Each row in a confusion matrix represents a true class and each column represents the instances in a predicted class. Let's look at an example:

|            | setosa | versicolor | virginica |
| ---------- | ------ | ---------- | --------- |
| setosa     | 13     | 0          | 0         |
| versicolor | 0      | 10         | 6         |
| virginica  | 0      | 0          | 9         |

What information does this confusion matrix provide?

* All setosa instances were properly predicted as such (true positives).
* The model always correctly classifies the setosa class (there are no false positives).
* 10 versicolor instances were properly classified, but 6 instances were misclassified as virginica.
* All virginica insances were properly classified as such.


## How to Use

At minimum, this metric requires predictions and references as inputs.

```python
>>> confusion_metric = evaluate.load("confusion_matrix")
>>> results = confusion_metric.compute(references=[0, 1, 1, 2, 0, 2, 2], predictions=[0, 2, 1, 1, 0, 2, 0])
>>> print(results)
{'confusion_matrix': [[2, 0, 0], [0, 1, 1], [1, 1, 1]]}
```


### Inputs
- **predictions** (`list` of `int`): Predicted labels.
- **references** (`list` of `int`): Ground truth labels.
- **labels** (`list` of `int`): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
- **sample_weight** (`list` of `float`): Sample weights.
- **normalize** (`str`): Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. (Valid values are `'pred', 'all', 'true'` or `None`).


### Output Values
- **confusion_matrix** (`list` of `list` of `str`): Confusion matrix. Minimum possible value is 0. Maximum possible value is 1.0, or the number of examples in the input, if `normalize` is set to `None` (default value).

Output Example(s):
```python
{'confusion_matrix': [[2, 0, 0], [0, 1, 1], [1, 1, 1]]}
```

This metric outputs a dictionary, containing the confusion matrix.


### Examples

Example 1 - A simple example

```python
>>> confusion_metric = evaluate.load("confusion_matrix")
>>> results = confusion_metric.compute(references=[0, 1, 1, 2, 0, 2, 2], predictions=[0, 2, 1, 1, 0, 2, 0])
>>> print(results)
{'confusion_matrix': [[2, 0, 0], [0, 1, 1], [1, 1, 1]]}
```

## Citation(s)
```bibtex
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
```


## Further References

* https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
* https://en.wikipedia.org/wiki/Confusion_matrix
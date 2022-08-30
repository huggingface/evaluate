---
title: Brier Score
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
  The Brier score is a measure of the error between two probability distributions.
---

# Metric Card for Brier Score


## Metric Description
Brier score is a type of evaluation metric for classification tasks, where you predict outcomes such as win/lose, spam/ham, click/no-click etc.
`BrierScore = 1/N * sum( (p_i - o_i)^2 )`

Where `p_i` is the prediction probability of occurrence of the event, and the term `o_i` is equal to 1 if the event occurred and 0 if not. Which means: the lower the value of this score, the better the prediction. 
## How to Use

At minimum, this metric requires predictions and references as inputs.

```python
>>> brier_score = evaluate.load("brier_score")
>>> predictions = np.array([0, 0, 1, 1])
>>> references = np.array([0.1, 0.9, 0.8, 0.3])
>>> results = brier_score.compute(predictions=predictions, references=references)
```

### Inputs

Mandatory inputs: 
- `predictions`: numeric array-like of shape (`n_samples,`) or (`n_samples`, `n_outputs`), representing the estimated target values.

- `references`: numeric array-like of shape (`n_samples,`) or (`n_samples`, `n_outputs`), representing the ground truth (correct) target values.

Optional arguments:
- `sample_weight`: numeric array-like of shape (`n_samples,`) representing sample weights. The default is `None`.
- `pos_label`: the label of the positive class. The default is `1`.
        

### Output Values
This metric returns a dictionary with the following keys:
- `brier_score (float)`: the computed Brier score.


Output Example(s):
```python
{'brier_score': 0.5}
```

#### Values from Popular Papers


### Examples
```python
>>> brier_score = evaluate.load("brier_score")
>>> predictions = np.array([0, 0, 1, 1])
>>> references = np.array([0.1, 0.9, 0.8, 0.3])
>>> results = brier_score.compute(predictions=predictions, references=references)
>>> print(results)
{'brier_score': 0.3375}
```
Example with `y_true` contains string, an error will be raised and `pos_label` should be explicitly specified.
```python
>>> brier_score_metric = evaluate.load("brier_score")
>>> predictions =  np.array(["spam", "ham", "ham", "spam"])
>>> references = np.array([0.1, 0.9, 0.8, 0.3])
>>> results = brier_score.compute(predictions, references, pos_label="ham")
>>> print(results) 
{'brier_score': 0.0374}
```
## Limitations and Bias
The [brier_score](https://huggingface.co/metrics/brier_score) is appropriate for binary and categorical outcomes that can be structured as true or false, but it is inappropriate for ordinal variables which can take on three or more values.
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

@Article{brier1950verification,
  title={Verification of forecasts expressed in terms of probability},
  author={Brier, Glenn W and others},
  journal={Monthly weather review},
  volume={78},
  number={1},
  pages={1--3},
  year={1950}
}
```
## Further References
- [Brier Score - Wikipedia](https://en.wikipedia.org/wiki/Brier_score)
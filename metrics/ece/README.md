---
title: Expected Calibration Error
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
  Expected Calibration Error (ECE) measures how well predicted confidence scores align with actual outcomes.
---

# Metric Card for Expected Calibration Error (ECE)

## Metric Description
Expected Calibration Error (ECE) measures how well a classifier's predicted confidence scores align with actual outcomes. Predictions are binned by confidence, and ECE is the weighted average of |accuracy - confidence| across bins. A perfectly calibrated model has ECE = 0.

## How to Use

At minimum, this metric requires predictions and references as inputs.

```python
>>> ece = evaluate.load("ece")
>>> references = np.array([0, 0, 1, 1])
>>> predictions = np.array([0.25, 0.25, 0.75, 0.75])
>>> results = ece.compute(references=references, predictions=predictions, n_bins=2)
>>> print(results)
{'ece': 0.25}
```

### Inputs

Mandatory inputs: 
- `references`: array-like of shape (`n_samples,`), representing the ground truth labels. Can be numeric (0/1 or -1/1) or strings.
- `predictions`: array-like of shape (`n_samples,`), representing the predicted confidence scores in [0, 1] for the positive class.

Optional arguments:
- `n_bins`: int, number of bins (default is 10).
- `equal_intervals`: bool, default=True. Use equal-width bins (True) or equal-mass/quantile bins (False).
- `pos_label`: int or str, default=None. Label of the positive class. Inferred automatically for numeric labels.

### Output Values
This metric returns a dictionary with:
- `ece` (float): Expected Calibration Error.

```python
{'ece': 0.25}
```

## Limitations and Bias
ECE is appropriate for binary classification with probabilistic predictions. The choice of `n_bins` affects the result; more bins yield noisier estimates. ECE requires predicted probabilities for the positive class only.

## Citation(s)
```bibtex
@inproceedings{guo2017calibration,
  title={On Calibration of Modern Neural Networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle={Proceedings of the 34th International Conference on Machine Learning},
  year={2017}
}
```

## Further References
- [On Calibration of Modern Neural Networks (Guo et al., 2017)](https://arxiv.org/abs/1706.04599)

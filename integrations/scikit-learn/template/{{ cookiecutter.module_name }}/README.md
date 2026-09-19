---
title: {{ cookiecutter.module_name }}
emoji: 🤗 
colorFrom: blue
colorTo: orange
tags:
- evaluate
- {{ cookiecutter.module_type }}
- sklearn
description: >-
  "{{ cookiecutter.docstring_first_line }}"
sdk: gradio
sdk_version: 3.12.0
app_file: app.py
pinned: false
---

This metric is part of the Scikit-learn integration into 🤗 Evaluate. You can find all available metrics in the [Scikit-learn organization](https://huggingface.co/scikit-learn) on the Hugging Face Hub.

<p align="center">
  <img src="https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/1280px-scikit-learn-logo.png" width="400"/>
</p>

# Metric Card for `sklearn.metrics.{{ cookiecutter.module_name }}`




## Input Convention

To be consistent with the `evaluate` input conventions the scikit-learn inputs are renamed:

- `{{cookiecutter.label_name}}`: `references`
- `{{cookiecutter.preds_name}}`: `predictions`


## Usage

```python
import evaluate

metric = evaluate.load("sklearn/{{ cookiecutter.module_name }}")
results = metric.compute(references=references, predictions=predictions)
```

## Description

{{ cookiecutter.docstring }}

## Citation
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
- Docs: {{ cookiecutter.docs_url }}
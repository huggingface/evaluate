---
title: BLEURT
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
  BLEURT a learnt evaluation metric for Natural Language Generation. It is built using multiple phases of transfer learning starting from a pretrained BERT model (Devlin et al. 2018)
  and then employing another pre-training phrase using synthetic data. Finally it is trained on WMT human annotations. You may run BLEURT out-of-the-box or fine-tune
  it for your specific application (the latter is expected to perform better).

  See the project's README at https://github.com/google-research/bleurt#readme for more information.
---

# Metric Card for BLEURT


## Metric Description
BLEURT is a learned evaluation metric for Natural Language Generation. It is built using multiple phases of transfer learning starting from a pretrained BERT model [Devlin et al. 2018](https://arxiv.org/abs/1810.04805), employing another pre-training phrase using synthetic data, and finally trained on WMT human annotations. 

It is possible to run BLEURT out-of-the-box or fine-tune it for your specific application (the latter is expected to perform better).
See the project's [README](https://github.com/google-research/bleurt#readme) for more information.

## Intended Uses
BLEURT is intended to be used for evaluating text produced by language models. 

## How to Use

This metric takes as input lists of predicted sentences and reference sentences:

```python
>>> predictions = ["hello there", "general kenobi"]
>>> references = ["hello there", "general kenobi"]
>>> bleurt = load("bleurt", module_type="metric")
>>> results = bleurt.compute(predictions=predictions, references=references)
```

### Inputs

For the `load` function:

- **config_name** (`str`): BLEURT checkpoint. Will default to `"bleurt-base-128"` if not specified. Other models that can be chosen are: `"bleurt-tiny-128"`, `"bleurt-tiny-512"`, `"bleurt-base-128"`, `"bleurt-base-512"`, `"bleurt-large-128"`, `"bleurt-large-512"`, `"BLEURT-20-D3"`, `"BLEURT-20-D6"`, `"BLEURT-20-D12"` and `"BLEURT-20"`.

For the `compute` function:

- **predictions** (`list` of `str`s): List of generated sentences to score.
- **references** (`list` of `str`s): List of references to compare to.

### Output Values
- **scores** : a `list` of scores, one per prediction. 

Output Example:
```python
{'scores': [1.0295498371124268, 1.0445425510406494]}

```

BLEURT's output is always a number between 0 and (approximately 1). This value indicates how similar the generated text is to the reference texts, with values closer to 1 representing more similar texts. 

#### Values from Popular Papers

The [original BLEURT paper](https://arxiv.org/pdf/2004.04696.pdf) reported that the metric is better correlated with human judgment compared to similar metrics such as BERT and BERTscore.

BLEURT is used to compare models across different asks (e.g. (Table to text generation)[https://paperswithcode.com/sota/table-to-text-generation-on-dart?metric=BLEURT]).

### Examples

Example with the default model (`"bleurt-base-128"`):
```python
>>> predictions = ["hello there", "general kenobi"]
>>> references = ["hello there", "general kenobi"]
>>> bleurt = load("bleurt", module_type="metric")
>>> results = bleurt.compute(predictions=predictions, references=references)
>>> print(results)
{'scores': [1.0295498371124268, 1.0445425510406494]}
```

Example with the full `"BLEURT-20"` model checkpoint:
```python
>>> predictions = ["hello there", "general kenobi"]
>>> references = ["hello there", "general kenobi"]
>>> bleurt = load("bleurt", module_type="metric", config_name="BLEURT-20")
>>> results = bleurt.compute(predictions=predictions, references=references)
>>> print(results)
{'scores': [1.015415906906128, 0.9985226988792419]}
```

## Limitations and Bias
The [original BLEURT paper](https://arxiv.org/pdf/2004.04696.pdf) showed that BLEURT correlates well with human judgment, but this depends on the model and language pair selected.

Furthermore, currently BLEURT only supports English-language scoring, given that it leverages models trained on English corpora. It may also reflect, to a certain extent, biases and correlations that were present in the model training data. 

Finally, calculating the BLEURT metric involves downloading the BLEURT model that is used to compute the score, which can take a significant amount of time depending on the model chosen. Starting with the default model, `bleurt-tiny`, and testing out larger models if necessary can be a useful approach if memory or internet speed is an issue.


## Citation
```bibtex
@inproceedings{bleurt,
  title={BLEURT: Learning Robust Metrics for Text Generation},
  author={Thibault Sellam and Dipanjan Das and Ankur P. Parikh},
  booktitle={ACL},
  year={2020},
  url={https://arxiv.org/abs/2004.04696}
}
```

## Further References
- The original [BLEURT GitHub repo](https://github.com/google-research/bleurt/)

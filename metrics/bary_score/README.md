---
title: Barry Score
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
  The Barry Score is a metric for evaluating the quality of a generative model. It is based on the idea that a good generative model should be able to generate a wide variety of outputs, and that the outputs should be diverse from each other. The Barry Score is a measure of the diversity of the outputs of a generative model.
---

# Metric Card for Barry Score
## Metric Description
BaryScore is a multi-layers metric based on pretrained contextualized representations. Similar to MoverScore, it aggregates the layers of Bert before computing a similarity score. By modelling the layer output of deep contextualized embeddings as a probability distribution rather than by a vector embedding; BaryScore aggregates the different outputs through the Wasserstein space topology. MoverScore (right) leverages the information available in other layers by aggregating the layers using a power mean and then use a Wasserstein distance ().

## How to Use
```python
>>> barry_score = evaluate.load("bary_score")
>>> references = ['I like my cakes very much','I hate these cakes!']
>>> predictions = ['I like my cakes very much','I like my cakes very much']
>>> results = barry_score.compute(predictions=predictions, references=references)
```

### Inputs
Mandatory inputs: 
- `predictions`: list of strings
- `references`: list of strings
### Output Values
This metric returns a dictionary with the following keys:
- `bary_score`: float

Output Example:
```python
{'baryscore_W': [2.220446049250313e-16, 0.4936737487362536], 'baryscore_SD_10': [0.9234963490510808, 1.0454139159538949], 'baryscore_SD_1': [0.7360736368883636, 0.9437504927697342], 'baryscore_SD_5': [0.9074753479358628, 1.036207173522238], 'baryscore_SD_0.1': [0.0007089180091671455, 0.5100520249124377], 'baryscore_SD_0.5': [0.4623988987972563, 0.8098911748431552], 'baryscore_SD_0.01': [2.220446049250317e-16, 0.49367374955620186], 'baryscore_SD_0.001': [2.220446049250313e-16, 3.118914392068233e-08]}
```
### Examples
```python
>>> bary_score = evaluate.load("bary_score")
>>> predictions = np.array([0, 0, 1, 1])
>>> references = np.array([0.1, 0.9, 0.8, 0.3])
>>> results = bary_score.compute(predictions=predictions, references=references)
>>> print(results)
{'baryscore_W': [2.220446049250313e-16, 0.4936737487362536], 'baryscore_SD_10': [0.9234963490510808, 1.0454139159538949], 'baryscore_SD_1': [0.7360736368883636, 0.9437504927697342], 'baryscore_SD_5': [0.9074753479358628, 1.036207173522238], 'baryscore_SD_0.1': [0.0007089180091671455, 0.5100520249124377], 'baryscore_SD_0.5': [0.4623988987972563, 0.8098911748431552], 'baryscore_SD_0.01': [2.220446049250317e-16, 0.49367374955620186], 'baryscore_SD_0.001': [2.220446049250313e-16, 3.118914392068233e-08]}
```

## Citation(s)
```bibtex
@inproceedings{colombo-etal-2021-automatic,
    title = "Automatic Text Evaluation through the Lens of {W}asserstein Barycenters",
    author = "Colombo, Pierre  and Staerman, Guillaume  and Clavel, Chlo{\'e}  and Piantanida, Pablo",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    year = "2021",
    pages = "10450--10466"
}
```

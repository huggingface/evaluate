---
title: BOLD
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
Bias in Open-ended Language Generation Dataset (BOLD) is a dataset to evaluate fairness  in open-ended language generation.
---

# Metric Card for BOLD

## Metric description


## How to use


## Output values

The output of the metric depends on the BOLD domain chosen, consisting of a dictionary that contains one or several of the following metrics:

`sentiment`:

`toxicity`:

`regard`:

`polarity`:

The `gender` and `race` domains return `sentiment`, `toxicity` and `regard`, the `religious_ideology` and `political_ideology` domains return `sentiment` and `toxicity`, and the `profession` domain returns `polarity`.


### Values from popular papers


## Examples

## Citation

```bibtex
@inproceedings{bold_2021,
author = {Dhamala, Jwala and Sun, Tony and Kumar, Varun and Krishna, Satyapriya and Pruksachatkun, Yada and Chang, Kai-Wei and Gupta, Rahul},
title = {BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation},
year = {2021},
isbn = {9781450383097},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3442188.3445924},
doi = {10.1145/3442188.3445924},
booktitle = {Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency},
pages = {862–872},
numpages = {11},
keywords = {natural language generation, Fairness},
location = {Virtual Event, Canada},
series = {FAccT '21}
}
```

## Further References

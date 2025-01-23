---
title: Summeval
emoji: 🌍
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.36.0
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: >-
  The SummEval dataset is a resource developed by the Yale LILY Lab and Salesforce Research for evaluating text summarization models. 
  It was created as part of a project to address shortcomings in summarization evaluation methods.
---

# Metric Card for SummEval Score

## Metric Description

The SummEval dataset is a resource developed by the Yale LILY Lab and Salesforce Research for evaluating text summarization models. 
It was created as part of a project to address shortcomings in summarization evaluation methods.

## How to Use 

1. **Loading the relevant SummEval metric** : the subsets of SummEval are the following: `rouge`, `rouge-we`, `mover-score`, `bert-score`, `summa-qa`, `blanc`, `supert`, `meteor`, `s3`, `data-stats`, `cider`, `chrf`, `bleu`, `syntactic`.

2. **Calculating the metric**: the metric takes two inputs : one list with the predictions of the model to score and one lists of references.

```python
from evaluate import load

summeval_metric = load('summeval', 'rouge')
predictions = ["hello there", "general kenobi"]
references = ["hello there", "general kenobi"]
results = summeval_metric.compute(predictions=predictions, references=references)
```

## Limitations and Bias

SummEval, like other evaluation frameworks, faces limitations such as reliance on possibly biased reference summaries, which can affect the accuracy of assessments. Automated metrics predominantly measure surface similarities, often missing deeper textual nuances like coherence and factual accuracy. Human evaluations introduce subjective biases and inconsistencies, while the framework’s effectiveness may also be limited by its focus on specific languages or domains. Moreover, the scalability challenges due to resource-intensive human evaluations can limit its broader applicability, particularly for those with limited resources.

## Citation

```bibtex
@article{fabbri2020summeval,
  title={SummEval: Re-evaluating Summarization Evaluation},
  author={Fabbri, Alexander R and Kry{\'s}ci{\'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
  journal={arXiv preprint arXiv:2007.12626},
  year={2020}
}
```
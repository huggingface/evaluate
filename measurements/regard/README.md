---
title: Regard
emoji: 🤗
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
tags:
- evaluate
- measurement
description: >-
Regard aims to measure language polarity towards and social perceptions of a demographic (e.g. gender, race, sexual orientation).
---

# Measurement Card for Regard

## Measurement Description

The `regard` measurement returns the estimated language polarity towards and social perceptions of a demographic (e.g. gender, race, sexual orientation).

It uses a model trained on labelled data from the paper ["The Woman Worked as a Babysitter: On Biases in Language Generation" (EMNLP 2019)](https://arxiv.org/abs/1909.01326)

## How to Use

This measurement requires a list of strings as input:

```python
>>> regard = evaluate.load("regard", module_type="measurement")
>>> input_texts = ["these girls are silly", "these boys are lost"]
>>> results = regard.compute(predictions=input_texts)
```

### Inputs
- **predictions** (list of `str`): A list of prediction/candidate sentences
- **aggregation** (`str`) (optional): determines the type of aggregation performed on the data.
    If set to `None`, the scores for each sentences are returned.
     Otherwise:
        - `average` : returns the average regard for each category (negative, positive, neutral, other)
        - `maximum`: returns the maximum regard for each category

### Output Values

By default, this measurement outputs a dictionary containing a list of regard scores, one for each sentence in `predictions`

```
{'regard': [[{'label': 'negative', 'score': 0.6691194772720337}, {'label': 'other', 'score': 0.22687028348445892}, {'label': 'neutral', 'score': 0.0852026417851448}, {'label': 'positive', 'score': 0.018807603046298027}], [{'label': 'neutral', 'score': 0.942646861076355}, {'label': 'positive', 'score': 0.02632979303598404}, {'label': 'negative', 'score': 0.020616641268134117}, {'label': 'other', 'score': 0.010406642220914364}]]}
```

With the `aggregation='maximum'` option, this measurement will output the maximum regard for each category (negative, positive, neutral, other):

```python
{'max_regard': {'negative': 0.6691194772720337, 'positive': 0.02632979303598404, 'neutral': 0.942646861076355, 'other': 0.22687028348445892}}
```

With the `aggregation='average'` option, this measurement will output the average regard for each category (negative, positive, neutral, other):

```python
{'average_regard': {'negative': 0.3448680592700839, 'positive': 0.022568698041141033, 'neutral': 0.5139247514307499, 'other': 0.11863846285268664}}
```

### Examples

Example 1 (default behavior):

```python
>>> regard = evaluate.load("regard", module_type="measurement")
>>> input_texts = ["these girls are silly", "these boys are lost"]
>>> results = regard.compute(predictions=input_texts)
>>> print(results)
{'regard': [[{'label': 'negative', 'score': 0.6691194772720337}, {'label': 'other', 'score': 0.22687028348445892}, {'label': 'neutral', 'score': 0.0852026417851448}, {'label': 'positive', 'score': 0.018807603046298027}], [{'label': 'neutral', 'score': 0.942646861076355}, {'label': 'positive', 'score': 0.02632979303598404}, {'label': 'negative', 'score': 0.020616641268134117}, {'label': 'other', 'score': 0.010406642220914364}]]}
```

Example 2 (returns the maximum toxicity score):
```python
>>> regard = evaluate.load("regard", module_type="measurement")
>>> input_texts = ["these girls are silly", "these boys are lost"]
>>> results = toxicity.compute(predictions=input_texts, aggregation = "maximum")
>>> print(results)
{'max_regard': {'negative': 0.6691194772720337, 'positive': 0.02632979303598404, 'neutral': 0.942646861076355, 'other': 0.22687028348445892}}
```

Example 3 (returns the average toxicity score):
```python
>>> regard = evaluate.load("regard", module_type="measurement")
>>> input_texts = ["these girls are silly", "these boys are lost"]
>>> results = toxicity.compute(predictions=input_texts, aggregation = "average")
>>> print(results)
{'average_regard': {'negative': 0.3448680592700839, 'positive': 0.022568698041141033, 'neutral': 0.5139247514307499, 'other': 0.11863846285268664}}
```

## Citation(s)
@article{https://doi.org/10.48550/arxiv.1909.01326,
  doi = {10.48550/ARXIV.1909.01326},
  url = {https://arxiv.org/abs/1909.01326},
  author = {Sheng, Emily and Chang, Kai-Wei and Natarajan, Premkumar and Peng, Nanyun},
  title = {The Woman Worked as a Babysitter: On Biases in Language Generation},
  publisher = {arXiv},
  year = {2019}
}


## Further References
- [`nlg-bias` library](https://github.com/ewsheng/nlg-bias/)

---
title: Anls
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 3.17.0
app_file: app.py
pinned: false
---
tags:
- evaluate
- metric
description: >-
  This metric wrap the official scoring script for version 1 of the Average Normalized Levenshtein Similarity (ANLS).

---

# Metric Card for ANLS

## Metric description
This metric wraps the official scoring script for version 1 of the Average Normalized Levenshtein Similarity (ANLS).

The ANLS smoothly captures the OCR mistakes applying a slight penalization in case of correct intended responses, but badly recognized. It also makes use of a threshold of value 0.5 that dictates whether the output of the metric will be the ANLS if its value is equal or bigger than 0.5 or 0 otherwise. The key point of this threshold is to determine if the answer has been correctly selected but not properly recognized, or on the contrary, the output is a wrong text selected from the options and given as an answer.

More formally, the ANLS between the net output and the ground truth answers is given by equation 1. Where N is the total number of questions, M total number of GT answers per question, a<sub>ij</sub> the ground truth answers where i = {0, ..., N}, and j = {0, ..., M}, and o<sub>qi</sub> be the network's answer for the i<sup>th</sup> question q<sub>i</sub>.

![alt text](https://rrc.cvc.uab.es/files/ANLS.png)

Reference: [Evaluation Metric](https://rrc.cvc.uab.es/?ch=11&com=tasks)

## How to use 
The metric takes two lists of question-answers dictionaries as inputs, one with the predictions of the model and the other with the references to be compared to.

_predictions_: List of question-answers dictionaries with the following key-values:

    - 'question_id': id of the question-answer pair as given in the references (see below)
    - 'prediction_text': the text of the answer

_references_: List of question-answers dictionaries with the following key-values:

    - 'question_id': id of the question-answer pair (see above)
    - 'answers': list of possible texts for the answer, as a list of strings

```python
from evaluate import load
squad_metric = load("anls")
results = anls_metric.compute(predictions=predictions, references=references)
```
## Output values

This metric outputs a dictionary with value 'anls_score' between 0.0 and 1.0

```
{'anls_score': 1.0}
```

## Examples 


```python
from evaluate import load
anls_metric = load("anls")
predictions = [{'question_id': '10285', 'prediction_text': 'Denver Broncos'},
                   {'question_id': '18601', 'prediction_text': '12/15/89'},
                   {'question_id': '16734', 'prediction_text': 'Dear dr. Lobo'}]

references = [{"answers": ["Denver Broncos", "Denver R. Broncos"], 'question_id': '10285'},
               {'answers': ['12/15/88'], 'question_id': '18601'},
               {'answers': ['Dear Dr. Lobo', 'Dr. Lobo'], 'question_id': '16734'}]
results = anls_metric.compute(predictions=predictions, references=references)
results
{'anls_metric': 1.0}
```


## Limitations and bias
This metric works only with datasets that have the same format as specified above.

## Considerations / Assumptions
As specified in website: [Tasks - Document Visual Question Answering](https://rrc.cvc.uab.es/?ch=17&com=tasks)

- Answers are not case sensitive
- Answers are space sensitive
- Answers or tokens comprising answers are not limited to a fixed size dictionary. It could be any word/token which is present in the document.

## Citation

    @article{,
    title = {Binary codes capable of correcting deletions, insertions, and reversals},
    journal = {Soviet physics doklady},
    volume = {10},
    number = {8},
    pages = {707--710},
    year = {1966},
    url = {https://nymity.ch/sybilhunting/pdf/Levenshtein1966a.pdf},
    author = {V. I. Levenshtein},
    
## Further References 

- [The Stanford Question Answering Dataset: Background, Challenges, Progress (blog post)](https://rajpurkar.github.io/mlx/qa-and-squad/)
- [Hugging Face Course -- Question Answering](https://huggingface.co/course/chapter7/7)

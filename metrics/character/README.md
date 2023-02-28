---
title: CharacTER
emoji: 🔤
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
tags:
- evaluate
- metric
- machine-translation
description: >-
  CharacTer is a character-level metric inspired by the commonly applied translation edit rate (TER).
---

# Metric Card for CharacTER

## Metric Description
CharacTer is a character-level metric inspired by the translation edit rate (TER) metric. It is 
defined as the minimum number of character edits required to adjust a hypothesis, until it completely matches the
reference, normalized by the length of the hypothesis sentence. CharacTer calculates the character level edit
distance while performing the shift edit on word level. Unlike the strict matching criterion in TER, a hypothesis
word is considered to match a reference word and could be shifted, if the edit distance between them is below a
threshold value. The Levenshtein distance between the reference and the shifted hypothesis sequence is computed on the
character level. In addition, the lengths of hypothesis sequences instead of reference sequences are used for
normalizing the edit distance, which effectively counters the issue that shorter translations normally achieve lower
TER.

## Intended Uses
CharacTER was developed for machine translation evaluation.

## How to Use

```python
import evaluate
character = evaluate.load("character")

# Single hyp/ref 
preds = ["this week the saudis denied information published in the new york times"]
refs = ["saudi arabia denied this week information published in the american new york times"]
results = character.compute(references=refs, predictions=preds)

# Corpus example
preds = ["this week the saudis denied information published in the new york times",
         "this is in fact an estimate"]
refs = ["saudi arabia denied this week information published in the american new york times",
        "this is actually an estimate"]
results = character.compute(references=refs, predictions=preds)
```

### Inputs
- **predictions**: a single prediction or a list of predictions to score. Each prediction should be a string with
     tokens separated by spaces.
- **references**: a single reference or a list of reference for each prediction. Each reference should be a string with
     tokens separated by spaces.


### Output Values

*=only when a list of references/hypotheses are given

- **count** (*): how many parallel sentences were processed
- **mean** (*): the mean CharacTER score
- **median** (*): the median score
- **std** (*): standard deviation of the score
- **min** (*): smallest score
- **max** (*): largest score
- **cer_scores**: all scores, one per ref/hyp pair

### Output Example
```python
{
    'count': 2,
    'mean': 0.3127282211789254,
    'median': 0.3127282211789254,
    'std': 0.07561653111280243,
    'min': 0.25925925925925924,
    'max': 0.36619718309859156,
    'cer_scores': [0.36619718309859156, 0.25925925925925924]
}
```

## Citation
```bibtex
@inproceedings{wang-etal-2016-character,
    title = "{C}harac{T}er: Translation Edit Rate on Character Level",
    author = "Wang, Weiyue  and
      Peter, Jan-Thorsten  and
      Rosendahl, Hendrik  and
      Ney, Hermann",
    booktitle = "Proceedings of the First Conference on Machine Translation: Volume 2, Shared Task Papers",
    month = aug,
    year = "2016",
    address = "Berlin, Germany",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W16-2342",
    doi = "10.18653/v1/W16-2342",
    pages = "505--510",
}
```

## Further References
- Repackaged version that is used in this HF implementation: [https://github.com/bramvanroy/CharacTER](https://github.com/bramvanroy/CharacTER)
- Original version: [https://github.com/rwth-i6/CharacTER](https://github.com/rwth-i6/CharacTER)

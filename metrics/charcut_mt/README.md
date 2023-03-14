---
title: CharCut
emoji: 🔤
colorFrom: blue
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
  CharCut is a character-based machine translation evaluation metric.
---

# Metric Card for CharacTER

## Metric Description
CharCut compares outputs of MT systems with reference translations. The matching algorithm is based on an iterative
search for longest common substrings, combined with a length-based threshold that limits short and noisy character
matches. As a similarity metric this is not new, but to the best of our knowledge it was never applied to highlighting
and scoring of MT outputs. It has the neat effect of keeping character-based differences readable by humans.

## Intended Uses
CharCut was developed for machine translation evaluation.

## How to Use

```python
import evaluate
charcut = evaluate.load("charcut")
preds = ["this week the saudis denied information published in the new york times",
                "this is in fact an estimate"]
refs = ["saudi arabia denied this week information published in the american new york times",
                "this is actually an estimate"]
results = charcut.compute(references=refs, predictions=preds)
print(results)
# {'charcut_mt': 0.1971153846153846}

```
### Inputs
- **predictions**: a single prediction or a list of predictions to score. Each prediction should be a string with
     tokens separated by spaces.
- **references**: a single reference or a list of reference for each prediction. Each reference should be a string with
     tokens separated by spaces.


### Output Values
- **charcut_mt**: the CharCut evaluation score (lower is better)

### Output Example
```python
{'charcut_mt': 0.1971153846153846}
```

## Citation
```bibtex
@inproceedings{lardilleux-lepage-2017-charcut,
    title = "{CHARCUT}: Human-Targeted Character-Based {MT} Evaluation with Loose Differences",
    author = "Lardilleux, Adrien  and
      Lepage, Yves",
    booktitle = "Proceedings of the 14th International Conference on Spoken Language Translation",
    month = dec # " 14-15",
    year = "2017",
    address = "Tokyo, Japan",
    publisher = "International Workshop on Spoken Language Translation",
    url = "https://aclanthology.org/2017.iwslt-1.20",
    pages = "146--153",
    abstract = "We present CHARCUT, a character-based machine translation evaluation metric derived from a human-targeted segment difference visualisation algorithm. It combines an iterative search for longest common substrings between the candidate and the reference translation with a simple length-based threshold, enabling loose differences that limit noisy character matches. Its main advantage is to produce scores that directly reflect human-readable string differences, making it a useful support tool for the manual analysis of MT output and its display to end users. Experiments on WMT16 metrics task data show that it is on par with the best {``}un-trained{''} metrics in terms of correlation with human judgement, well above BLEU and TER baselines, on both system and segment tasks.",
}
```

## Further References
- Repackaged version that is used in this HF implementation: [https://github.com/BramVanroy/CharCut](https://github.com/BramVanroy/CharCut)
- Original version: [https://github.com/alardill/CharCut](https://github.com/alardill/CharCut)

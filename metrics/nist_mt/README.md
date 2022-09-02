---
title: NIST
emoji: 🤗 
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: 
  TODO
---

# Metric Card for BLEU


## Metric Description
TODO

## Intended Uses
NIST was developed for machine translation evaluation.

## How to Use

```python
>>> nist = evaluate.load("nist_mt")
>>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
...               'ensures', 'that', 'the', 'military', 'always',
...               'obeys', 'the', 'commands', 'of', 'the', 'party']

>>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
...               'ensures', 'that', 'the', 'military', 'will', 'forever',
...               'heed', 'Party', 'commands']

>>> reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which',
...               'guarantees', 'the', 'military', 'forces', 'always',
...               'being', 'under', 'the', 'command', 'of', 'the',
...               'Party']

>>> reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
...               'army', 'always', 'to', 'heed', 'the', 'directions',
...               'of', 'the', 'party']

>>> nist.compute(hypothesis1, [reference1, reference2])
{'nist_mt': 3.3709935957649324}
```

### Inputs
- **predictions**: tokenized predictions to score. For sentence-level NIST, a list of tokens (str);
     for corpus-level NIST, a list (sentences) of lists of tokens (str)
- **references**:  potentially multiple tokenized references for each prediction.  For sentence-level NIST, a
     list (multiple potential references) of list of tokens (str); for corpus-level NIST, a list (corpus) of lists
     (multiple potential references) of lists of tokens (str)
- **n**: highest n-gram order

### Output Values
- **nist** (`float`): nist score

Output Example:
```python
{'nist_mt': 3.3709935957649324}
```


#### Values from Popular Papers
TODO

### Examples

TODO



## Citation
```bibtex
@inproceedings{10.5555/1289189.1289273,
    author = {Doddington, George},
    title = {Automatic Evaluation of Machine Translation Quality Using N-Gram Co-Occurrence Statistics},
    year = {2002},
    publisher = {Morgan Kaufmann Publishers Inc.},
    address = {San Francisco, CA, USA},
    abstract = {Evaluation is recognized as an extremely helpful forcing function in Human Language Technology R&amp;D. Unfortunately, evaluation has not been a very powerful tool in machine translation (MT) research because it requires human judgments and is thus expensive and time-consuming and not easily factored into the MT research agenda. However, at the July 2001 TIDES PI meeting in Philadelphia, IBM described an automatic MT evaluation technique that can provide immediate feedback and guidance in MT research. Their idea, which they call an "evaluation understudy", compares MT output with expert reference translations in terms of the statistics of short sequences of words (word N-grams). The more of these N-grams that a translation shares with the reference translations, the better the translation is judged to be. The idea is elegant in its simplicity. But far more important, IBM showed a strong correlation between these automatically generated scores and human judgments of translation quality. As a result, DARPA commissioned NIST to develop an MT evaluation facility based on the IBM work. This utility is now available from NIST and serves as the primary evaluation measure for TIDES MT research.},
    booktitle = {Proceedings of the Second International Conference on Human Language Technology Research},
    pages = {138–145},
    numpages = {8},
    location = {San Diego, California},
    series = {HLT '02}
}
```

## Further References

This Hugging Face implementation uses [this NLTK implementation](https://github.com/nltk/nltk/blob/develop/nltk/translate/nist_score.py)

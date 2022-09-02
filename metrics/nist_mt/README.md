---
title: NIST_MT
emoji: 🤗 
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: 
  DARPA commissioned NIST to develop an MT evaluation facility based on the BLEU score.
---

# Metric Card for NIST's MT metric


## Metric Description
DARPA commissioned NIST to develop an MT evaluation facility based on the BLEU
score. The official script used by NIST to compute BLEU and NIST score is
mteval-14.pl. The main differences are:

 - BLEU uses geometric mean of the ngram overlaps, NIST uses arithmetic mean.
 - NIST has a different brevity penalty
 - NIST score from mteval-14.pl has a self-contained tokenizer (in the Hugging Face implementation we rely on NLTK's 
implementation of the NIST-specific tokenizer)

## Intended Uses
NIST was developed for machine translation evaluation.

## How to Use

```python
nist_mt = evaluate.load("nist_mt")
hypothesis1 = "It is a guide to action which ensures that the military always obeys the commands of the party"
reference1 = "It is a guide to action that ensures that the military will forever heed Party commands"
reference2 = "It is the guiding principle which guarantees the military forces always being under the command of the Party"
nist_mt.compute(hypothesis1, [reference1, reference2])
# {'nist_mt': 3.3709935957649324}
```

### Inputs
- **predictions**: tokenized predictions to score. For sentence-level NIST, a list of tokens (str);
     for corpus-level NIST, a list (sentences) of lists of tokens (str)
- **references**:  potentially multiple tokenized references for each prediction.  For sentence-level NIST, a
     list (multiple potential references) of list of tokens (str); for corpus-level NIST, a list (corpus) of lists
     (multiple potential references) of lists of tokens (str)
- **n**: highest n-gram order
- **tokenize_kwargs**: arguments passed to the tokenizer (see: https://github.com/nltk/nltk/blob/90fa546ea600194f2799ee51eaf1b729c128711e/nltk/tokenize/nist.py#L139)

### Output Values
- **nist_mt** (`float`): NIST score

Output Example:
```python
{'nist_mt': 3.3709935957649324}
```


## Citation
```bibtex
@inproceedings{10.5555/1289189.1289273,
    author = {Doddington, George},
    title = {Automatic Evaluation of Machine Translation Quality Using N-Gram Co-Occurrence Statistics},
    year = {2002},
    publisher = {Morgan Kaufmann Publishers Inc.},
    address = {San Francisco, CA, USA},
    booktitle = {Proceedings of the Second International Conference on Human Language Technology Research},
    pages = {138–145},
    numpages = {8},
    location = {San Diego, California},
    series = {HLT '02}
}
```

## Further References

This Hugging Face implementation uses [the NLTK implementation](https://github.com/nltk/nltk/blob/develop/nltk/translate/nist_score.py)

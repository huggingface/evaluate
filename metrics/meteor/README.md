---
title: METEOR
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
  METEOR, an automatic metric for machine translation evaluation
  that is based on a generalized concept of unigram matching between the
  machine-produced translation and human-produced reference translations.
  Unigrams can be matched based on their surface forms, stemmed forms,
  and meanings; furthermore, METEOR can be easily extended to include more
  advanced matching strategies. Once all generalized unigram matches
  between the two strings have been found, METEOR computes a score for
  this matching using a combination of unigram-precision, unigram-recall, and
  a measure of fragmentation that is designed to directly capture how
  well-ordered the matched words in the machine translation are in relation
  to the reference.
  
  METEOR gets an R correlation value of 0.347 with human evaluation on the Arabic
  data and 0.331 on the Chinese data. This is shown to be an improvement on
  using simply unigram-precision, unigram-recall and their harmonic F1
  combination.
---

# Metric Card for METEOR

## Metric description

METEOR (Metric for Evaluation of Translation with Explicit ORdering) is a machine translation evaluation metric, which is calculated based on the harmonic mean of precision and recall, with recall weighted more than precision. 

METEOR is based on a generalized concept of unigram matching between the machine-produced translation and human-produced reference translations. Unigrams can be matched based on their surface forms, stemmed forms, and meanings. Once all generalized unigram matches between the two strings have been found, METEOR computes a score for this matching using a combination of unigram-precision, unigram-recall, and a measure of fragmentation that is designed to directly capture how well-ordered the matched words in the machine translation are in relation to the reference. 


## How to use 

METEOR has two mandatory arguments:

`predictions`: a `list` of predictions to score. Each prediction should be a string with tokens separated by spaces.

`references`: a `list` of references (in the case of one `reference` per `prediction`), or a `list` of `lists` of references (in the case of multiple `references` per `prediction`. Each reference should be a string with tokens separated by spaces.

It also has several optional parameters:

`alpha`: Parameter for controlling relative weights of precision and recall. The default value is `0.9`.

`beta`: Parameter for controlling shape of penalty as a function of fragmentation. The default value is `3`.

`gamma`: The relative weight assigned to fragmentation penalty. The default is `0.5`. 

Refer to the [METEOR paper](https://aclanthology.org/W05-0909.pdf) for more information about parameter values and ranges.

```python
>>> meteor = evaluate.load('meteor')
>>> predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
>>> references = ["It is a guide to action that ensures that the military will forever heed Party commands"]
>>> results = meteor.compute(predictions=predictions, references=references)
```

## Output values

The metric outputs a dictionary containing the METEOR score. Its values range from 0 to 1, e.g.:
```
{'meteor': 0.9999142661179699}
```


### Values from popular papers
The [METEOR paper](https://aclanthology.org/W05-0909.pdf) does not report METEOR score values for different models, but it does report that METEOR gets an R correlation value of 0.347 with human evaluation on the Arabic data and 0.331 on the Chinese data. 


## Examples 

One `reference` per `prediction`:

```python
>>> meteor = evaluate.load('meteor')
>>> predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
>>> reference = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
>>> results = meteor.compute(predictions=predictions, references=reference)
>>> print(round(results['meteor'], 2))
1.0
```

Multiple `references` per `prediction`:

```python
>>> meteor = evaluate.load('meteor')
>>> predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
>>> references = [['It is a guide to action that ensures that the military will forever heed Party commands', 'It is the guiding principle which guarantees the military forces always being under the command of the Party', 'It is the practical guide for the army always to heed the directions of the party']]
>>> results = meteor.compute(predictions=predictions, references=references)
>>> print(round(results['meteor'], 2))
1.0
```

Multiple `references` per `prediction`, partial match:

```python
>>> meteor = evaluate.load('meteor')
>>> predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
>>> references = [['It is a guide to action that ensures that the military will forever heed Party commands', 'It is the guiding principle which guarantees the military forces always being under the command of the Party', 'It is the practical guide for the army always to heed the directions of the party']]
>>> results = meteor.compute(predictions=predictions, references=references)
>>> print(round(results['meteor'], 2))
0.69
```

## Limitations and bias

While the correlation between METEOR and human judgments was measured for Chinese and Arabic and found to be significant, further experimentation is needed to check its correlation for other languages. 

Furthermore, while the alignment and matching done in METEOR is based on unigrams, using multiple word entities (e.g. bigrams) could contribute to improving its accuracy -- this has been proposed in [more recent publications](https://www.cs.cmu.edu/~alavie/METEOR/pdf/meteor-naacl-2010.pdf) on the subject.


## Citation

```bibtex
@inproceedings{banarjee2005,
  title     = {{METEOR}: An Automatic Metric for {MT} Evaluation with Improved Correlation with Human Judgments},
  author    = {Banerjee, Satanjeev  and Lavie, Alon},
  booktitle = {Proceedings of the {ACL} Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization},
  month     = jun,
  year      = {2005},
  address   = {Ann Arbor, Michigan},
  publisher = {Association for Computational Linguistics},
  url       = {https://www.aclweb.org/anthology/W05-0909},
  pages     = {65--72},
}
```
    
## Further References 
- [METEOR -- Wikipedia](https://en.wikipedia.org/wiki/METEOR)
- [METEOR score -- NLTK](https://www.nltk.org/_modules/nltk/translate/meteor_score.html)


---
title: poseval
emoji: 🤗 
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: >-
  The poseval metric can be used to evaluate POS taggers. Since seqeval does not work well with POS data 
  that is not in IOB format the poseval is an alternative. It treats each token in the dataset as independant 
  observation and computes the precision, recall and F1-score irrespective of sentences. It uses scikit-learns's
  classification report to compute the scores.
---

# Metric Card for peqeval

## Metric description

The poseval metric can be used to evaluate POS taggers. Since seqeval does not work well with POS data (see e.g. [here](https://stackoverflow.com/questions/71327693/how-to-disable-seqeval-label-formatting-for-pos-tagging)) that is not in IOB format the poseval is an alternative. It treats each token in the dataset as independant observation and computes the precision, recall and F1-score irrespective of sentences. It uses scikit-learns's [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) to compute the scores.


## How to use 

Poseval produces labelling scores along with its sufficient statistics from a source against references.

It takes two mandatory arguments:

`predictions`: a list of lists of predicted labels, i.e. estimated targets as returned by a tagger.

`references`: a list of lists of reference labels, i.e. the ground truth/target values.

It can also take several optional arguments:

`zero_division`: Which value to substitute as a metric value when encountering zero division. Should be one of [`0`,`1`,`"warn"`]. `"warn"` acts as `0`, but the warning is raised.


```python
>>> predictions = [['INTJ', 'ADP', 'PROPN', 'NOUN', 'PUNCT', 'INTJ', 'ADP', 'PROPN', 'VERB', 'SYM']]
>>> references = [['INTJ', 'ADP', 'PROPN', 'PROPN', 'PUNCT', 'INTJ', 'ADP', 'PROPN', 'PROPN', 'SYM']]
>>> poseval = evaluate.load("poseval")
>>> results = poseval.compute(predictions=predictions, references=references)
>>> print(list(results.keys()))
['ADP', 'INTJ', 'NOUN', 'PROPN', 'PUNCT', 'SYM', 'VERB', 'accuracy', 'macro avg', 'weighted avg']
>>> print(results["accuracy"])
0.8
>>> print(results["PROPN"]["recall"])
0.5
```

## Output values

This metric returns a a classification report as a dictionary with a summary of scores for overall and per type:

Overall (weighted and macro avg):

`accuracy`: the average [accuracy](https://huggingface.co/metrics/accuracy), on a scale between 0.0 and 1.0.
    
`precision`: the average [precision](https://huggingface.co/metrics/precision), on a scale between 0.0 and 1.0.
    
`recall`: the average [recall](https://huggingface.co/metrics/recall), on a scale between 0.0 and 1.0.

`f1`: the average [F1 score](https://huggingface.co/metrics/f1), which is the harmonic mean of the precision and recall. It also has a scale of 0.0 to 1.0.

Per type (e.g. `MISC`, `PER`, `LOC`,...):

`precision`: the average [precision](https://huggingface.co/metrics/precision), on a scale between 0.0 and 1.0.

`recall`: the average [recall](https://huggingface.co/metrics/recall), on a scale between 0.0 and 1.0.

`f1`: the average [F1 score](https://huggingface.co/metrics/f1), on a scale between 0.0 and 1.0.


## Examples 

```python
>>> predictions = [['INTJ', 'ADP', 'PROPN', 'NOUN', 'PUNCT', 'INTJ', 'ADP', 'PROPN', 'VERB', 'SYM']]
>>> references = [['INTJ', 'ADP', 'PROPN', 'PROPN', 'PUNCT', 'INTJ', 'ADP', 'PROPN', 'PROPN', 'SYM']]
>>> poseval = evaluate.load("poseval")
>>> results = poseval.compute(predictions=predictions, references=references)
>>> print(list(results.keys()))
['ADP', 'INTJ', 'NOUN', 'PROPN', 'PUNCT', 'SYM', 'VERB', 'accuracy', 'macro avg', 'weighted avg']
>>> print(results["accuracy"])
0.8
>>> print(results["PROPN"]["recall"])
0.5
```

## Limitations and bias

In contrast to [seqeval](https://github.com/chakki-works/seqeval), the poseval metric treats each token independently and computes the classification report over all concatenated sequences..


## Citation

```bibtex
@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}
```
    
## Further References 
- [README for seqeval at GitHub](https://github.com/chakki-works/seqeval)
- [Classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) 
- [Issues with seqeval](https://stackoverflow.com/questions/71327693/how-to-disable-seqeval-label-formatting-for-pos-tagging)
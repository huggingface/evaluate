---
title: ANLS
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
  ANLS (The average normalized Levenshtein similarity) is a string similarity metric that measures the difference between two strings. It is based on the Levenshtein distance, which is a measure of the minimum number of edit operations (insertions, deletions, and substitutions) needed to transform one string into another.
 
  The average normalized Levenshtein similarity score is calculated by normalizing the Levenshtein distance between two strings and transforming it into a similarity score that ranges from 0 to 1. 
---



## Metric Description
 ANLS (The average normalized Levenshtein similarity) is a string similarity metric that measures the difference between two strings. It is based on the Levenshtein distance, which is a measure of the minimum number of edit operations (insertions, deletions, and substitutions) needed to transform one string into another.

## Intended Uses
ANLS metric is most often used for speech recognition systems to match the transcribed speech to the reference transcript.

## How to Use

This metric takes as input a list of predicted sentences and a list of lists of reference sentences (since each predicted sentence can have multiple references):

```python
>>> references = [
...       ["hello my friend", "friend", "hi"], 
...       ["leave"], 
...       ["bad","okay"]
...               ]
>>> predictions = ["hi friend", "I am leaving", "that's good"]
>>> score = average_normalized_Levenshtein_similarity(references, predictions)
>>> print(score)
0.73333  
```

### Inputs
- **targets** (`list` of `list`s of `str`s).
- **references** (`list` of `str`s)



### Output Values
- **anls** (`float`): ANLS score

Output Example:
```python
0.73333  
```

ANLS's output is always a number between 0 and 1. This value indicates how similar the predictions is to the targets texts, with values closer to 1 representing more similar texts.



### Examples

Example where each prediction has 1 reference:
```python
>>> predictions = ["A quick brown fox jumps over the sleeping hound."]
>>> references = [
...  ["The quick brown fox jumps over the lazy dog."]
...  ]
>>> anls = evaluate.load("anls")
>>> results = anls.compute(predictions=predictions, references=references)
>>> print(results)
0.75

```

Example where the second prediction has 3 targets:
```python
>>> references = [
...       ["hello my friend", "friend", "hi"], 
...       ["leave"], 
...       ["bad","okay"]
...               ]
>>> predictions = ["hi friend", "I am leaving", "that's good"]
>>> anls = evaluate.load("anls")
>>> results = anls.compute(predictions=predictions, references=references)
>>> print(results)
0.73333  

```

## Limitations and Bias
The average normalized Levenshtein similarity is a simple and widely used string similarity metric, but it has some limitations and potential biases that should be taken into account when using it:
- Limitation of edit operations: The Levenshtein distance only considers three types of edit operations (insertions, deletions, and substitutions) and does not take into account more complex transformations, such as transpositions or substitutions of multiple characters.
- Bias towards longer strings: The average normalized Levenshtein similarity has a bias towards longer strings, as the Levenshtein distance is divided by the length of the longer string. This can lead to results that do not accurately reflect the true similarity between strings.
- Bias towards common characters: The average normalized Levenshtein similarity may have a bias towards strings that have more common characters, as the Levenshtein distance will be lower for strings that have more characters in common.
- Sensitivity to order: The Levenshtein distance and average normalized Levenshtein similarity are sensitive to the order of characters in the strings being compared, which may not always reflect the true similarity between strings.
- Lack of semantic understanding: The average normalized Levenshtein similarity does not take into account the semantic meaning of the strings being compared, and is only based on the surface-level similarity between the strings.
- These limitations and biases should be taken into account when using the average normalized Levenshtein similarity, and alternative string similarity metrics may be more appropriate for specific use cases.

## Citation
```bibtex
@INPROCEEDINGS{ANLS,
    author = {Vladimir Levenshtein},
    title = {Binary Codes Capable of Correcting Deletions, Insertions, and Reversals},
    booktitle = {},
    year = {1965},
    pages = {}
}
```

## Further References


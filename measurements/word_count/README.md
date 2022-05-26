---
title: Word Count
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
---

# Measurement Card for Word Count

## Measurement Description

The `word_count` measurement returns the total number of word count of the input string, using the sklearn's [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

## How to Use

This measurement requires a list of strings as input:

```python
>>> data = ["hello world and hello moon"]
>>> wordcount= evaluate.load("word_count")
>>> results = wordcount.compute(data=data)
```

### Inputs
- **data** (list of `str`): The input list of strings for which the word length is calculated.
- **max_vocab** (`int`): (optional) the top number of words to consider (can be specified if dataset is too large)

### Output Values
- **total_word_count** (`float`): the total number of words in the input string(s).
- **unique_words** (`float`): the number of unique words in the input string(s).

Output Example(s):

```python
{'total_word_count': 5, 'unique_words': 4}


### Examples

Example for a single string

```python
>>> data = ["hello sun and goodbye moon"]
>>> wordcount = evaluate.load("word_count")
>>> results = wordcount.compute(data=data)
>>> print(results)
{'total_word_count': 5, 'unique_words': 5}
```

Example for a multiple strings
```python
>>> data = ["hello sun and goodbye moon", "foo bar foo bar"]
>>> wordcount = evaluate.load("word_count")
>>> results = wordcount.compute(data=data)
>>> print(results)
{'total_word_count': 9, 'unique_words': 7}
```

Example for a dataset from 🤗 Datasets:

```python
>>> imdb = datasets.load_dataset('imdb', split = 'train')
>>> wordcount = evaluate.load("word_count")
>>> results = wordcount.compute(data=imdb['text'])
>>> print(results)
{'total_word_count': 5678573, 'unique_words': 74849}
```

## Citation(s)


## Further References
- [Sklearn `CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

---
title: Word Length
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
  Returns the average length (in terms of the number of words) of the input data.
---

# Measurement Card for Word Length


## Measurement Description

The `word_length` measurement returns the average word count of the input strings, based on tokenization using [NLTK word_tokenize](https://www.nltk.org/api/nltk.tokenize.html).

## How to Use

This measurement requires a list of strings as input:

```python
>>> data = ["hello world"]
>>> wordlength = evaluate.load("word_length", module_type="measurement")
>>> results = wordlength.compute(data=data)
```

### Inputs
- **data** (list of `str`): The input list of strings for which the word length is calculated.
- **tokenizer** (`Callable`) : approach used for tokenizing `data` (optional). The default tokenizer is [NLTK's `word_tokenize`](https://www.nltk.org/api/nltk.tokenize.html). This can be replaced by any function that takes a string as input and returns a list of tokens as output.

### Output Values
- **average_word_length**(`float`): the average number of words in the input string(s).

Output Example(s):

```python
{"average_word_length": 245}
```

This metric outputs a dictionary containing the number of words in the input string (`word length`).

### Examples

Example for a single string

```python
>>> data = ["hello sun and goodbye moon"]
>>> wordlength = evaluate.load("word_length", module_type="measurement")
>>> results = wordlength.compute(data=data)
>>> print(results)
{'average_word_length': 5}
```

Example for a multiple strings
```python
>>> data = ["hello sun and goodbye moon", "foo bar foo bar"]
>>> wordlength = evaluate.load("word_length", module_type="measurement")
>>> results = wordlength.compute(data=text)
{'average_word_length': 4.5}
```

## Citation(s)


## Further References
- [NLTK's `word_tokenize`](https://www.nltk.org/api/nltk.tokenize.html)

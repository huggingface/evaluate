---
title: Text Duplicates
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
  Returns the duplicate fraction of duplicate strings in the input.
---

# Measurement Card for Text Duplicates

## Measurement Description

The `text_duplicates` measurement returns the fraction of duplicated strings in the input data.

## How to Use

This measurement requires a list of strings as input:

```python
>>> data = ["hello sun","hello moon", "hello sun"]
>>> duplicates = evaluate.load("text_duplicates")
>>> results = duplicates.compute(data=data)
```

### Inputs
- **data** (list of `str`): The input list of strings for which the duplicates are calculated.

### Output Values
- **duplicate_fraction**(`float`): the fraction of duplicates in the input string(s).
- **duplicates_list**(`list`): (optional) a list of tuples with the duplicate strings and the number of times they are repeated.

By default, this measurement outputs a dictionary containing the fraction of duplicates in the input string(s) (`duplicate_fraction`):
  )
```python
{'duplicate_fraction': 0.33333333333333337}
```

With the `list_duplicates=True` option, this measurement will also output a dictionary of tuples with duplicate strings and their counts.

```python
{'duplicate_fraction': 0.33333333333333337, 'duplicates_list': {'hello sun': 2}}
```

Warning: the `list_duplicates=True` function can be memory-intensive for large datasets.

### Examples

Example with no duplicates

```python
>>> data = ["foo", "bar", "foobar"]
>>> duplicates = evaluate.load("text_duplicates")
>>> results = duplicates.compute(data=data)
>>> print(results)
{'duplicate_fraction': 0.0}
```

Example with multiple duplicates and `list_duplicates=True`:
```python
>>> data = ["hello sun", "goodbye moon", "hello sun", "foo bar", "foo bar"]
>>> duplicates = evaluate.load("text_duplicates")
>>> results = duplicates.compute(data=data)
>>> print(results)
{'duplicate_fraction': 0.4, 'duplicates_list': {'hello sun': 2, 'foo bar': 2}}
```

## Citation(s)


## Further References
- [`hashlib` library](https://docs.python.org/3/library/hashlib.html)

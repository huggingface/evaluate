---
title: Honest
emoji: 🤗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
tags:
- evaluate
- measurement
description: >-
The HONEST score aims to measure hurtful sentence completions in language models.
---

# Measurement Card for HONEST

## Measurement description
The HONEST score aims to measure hurtful sentence completions in language models.
The score uses HurtLex, a multilingual lexicon of hurtful language, to evaluate the completions.
It aims to quantify how often sentences are completed with a hurtful word, and if there is a difference between
groups (e.g. genders, sexual orientations, etc.).

## How to use

When loading the model, specify the language of the prompts and completions.
The available languages are: 'it' (Italian), 'fr' (French), 'es' (Spanish), 'pt' (Portuguese), 'ro' (Romanian), 'en' (English).
```python
>>> honest = evaluate.load('measurements/honest', 'en')
```
Arguments:
    **predictions** (list of list of `str`): a list of completions to [HONEST prompts](https://huggingface.co/datasets/MilaNLProc/honest)
    **groups** (list of `str`) (*optional*): a list of the identity groups each list of completions belongs to.


## Output values

`honest_score`: the HONEST score, representing the average number of hurtful completions across all groups
`honest_score_per_group`: the HONEST score of each group separately.

### Values from popular papers
In the [original HONEST paper](https://aclanthology.org/2021.naacl-main.191.pdf), the following scores were calculated for models, with Top K referring to the number of model completions that were evaluated:


| Model Name       | Top K =1 | Top K =5 |Top K =20 |
| ---------------- | -------- | -------- | -------- |
| UmBERTo (OSCAR)  | 5.24     | 8.19     |  7.14    |
| UmBERTo (Wiki)   | 5.48     | 7.19     |  5.14    |
| GilBERTo         | 7.14     | 11.57    |  8.68    |
| ItalianBERT XXL  | 9.05     | 10.67    |  9.12    |
| FlauBERT         | 4.76     | 3.29     |  2.43    |
| CamemBERT (OSCAR)| 18.57    | 9.62     |  7.07    |
| CamemBERT (Wiki) | 7.62     | 4.90     |  4.19    |
| BETO             | 4.29     |  5.95    |  6.88    |
| BERTimbau        | 4.05     | 6.00     |  5.04    |
| RomanianBERT     | 4.76     | 3.90     |  4.61    |
| BERT-base        | 1.19     | 2.67     |  3.55    |
| BERT-large       | 3.33     | 3.43     |  4.30    |
| RoBERTa-base     | 2.38     | 5.38     |  5.74    |
| RoBERTa-large    | 2.62     | 2.33     |  3.05    |
| DistilBERT-base  | 1.90     | 3.81     |  3.96    |
| GPT-2 (IT)       | 12.86    | 11.76    |  12.56   |
| GPT-2 (FR)       | 19.76    | 19.67    |  17.81   |
| GPT-2 (PT)       | 9.52     | 10.71    | 10.29    |
| GPT-2 (EN)       | 17.14    | 12.81    |  13.00   |


## Examples

Example 1: Calculating HONEST without groups

```python
>>> honest = evaluate.load('measurements/honest', lang = 'en')
>>> completions = [['CEO', 'businessman', 'politician'], ['good', 'excellent', 'smart'], ['secretary', 'prostitute', 'maid'], ['beautiful', 'smart', 'tall']]
>>> result = honest.compute(predictions=completions)
>>> round(result['honest_score'],3)
0.167
```

Example 2: Calculating HONEST with 2 groups (male/female)
```python
>>> honest = evaluate.load('measurements/honest', lang = 'en')
>>> completions = [['CEO', 'businessman', 'father'], ['good', 'excellent', 'smart'], ['secretary', 'prostitute', 'maid'], ['beautiful', 'smart', 'tall']]
>>> groups = ["male", "male", "female", "female"]
>>> results = honest.compute(predictions=completions, groups = groups)
>>> {g:round(s,3) for g,s in result['honest_score_per_group'].items()}
{'male': 0.0, 'female': 0.167}
```

Example 2: Calculating HONEST in another language (French)
```python
>>> honest = evaluate.load('measurements/honest', lang = 'fr')
>>> completions = [['PDG', 'avocat', 'père'], ['bon','excellent', 'intelligent'], ['secrétaire', 'prostituée', 'bonne'], ['belle', 'intelligente', 'grande']]
>>> result = honest.compute(predictions=completions)
>>> round(result['honest_score'],3)
0.083
```



## Citation

```bibtex
@inproceedings{nozza-etal-2021-honest,
    title = {"{HONEST}: Measuring Hurtful Sentence Completion in Language Models"},
    author = "Nozza, Debora and Bianchi, Federico  and Hovy, Dirk",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.191",
    doi = "10.18653/v1/2021.naacl-main.191",
    pages = "2398--2406",
}
```

```bibtex
@inproceedings{nozza-etal-2022-measuring,
    title = {Measuring Harmful Sentence Completion in Language Models for LGBTQIA+ Individuals},
    author = "Nozza, Debora and Bianchi, Federico and Lauscher, Anne and Hovy, Dirk",
    booktitle = "Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion",
    publisher = "Association for Computational Linguistics",
    year={2022}
}
```

## Further References
- Bassignana, Elisa, Valerio Basile, and Viviana Patti. ["Hurtlex: A multilingual lexicon of words to hurt."](http://ceur-ws.org/Vol-2253/paper49.pdf) 5th Italian Conference on Computational Linguistics, CLiC-it 2018. Vol. 2253. CEUR-WS, 2018.

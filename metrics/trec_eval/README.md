---
title: TREC Eval
datasets:
-  
tags:
- evaluate
- metric
sdk: gradio
sdk_version: 3.0.2
app_file: app.py
pinned: false
---

# Metric Card for TREC Eval

## Metric Description

The TREC Eval metric combines a number of information retrieval metrics such as precision and normalized Discounted Cumulative Gain (nDCG). It is used to score rankings of retrieved documents with reference values.

## How to Use
```Python
from evaluate import load
trec_eval = load("trec_eval")
results = trec_eval.compute(predictions=[run], references=[qrel])
```

### Inputs
- **predictions** *(dict): a single retrieval run.*
    - **query** *(int): Query ID.*
    - **q0** *(str): Literal `"q0"`.*
    - **docid** *(str): Document ID.*
    - **rank** *(int): Rank of document.*
    - **score** *(float): Score of document.*
    - **system** *(str): Tag for current run.*
- **references** *(dict): a single qrel.*
    - **query** *(int): Query ID.*
    - **q0** *(str): Literal `"q0"`.*
    - **docid** *(str): Document ID.*
    - **rel** *(int): Relevance of document.*

### Output Values
- **runid** *(str): Run name.*  
- **num_ret** *(int): Number of retrieved documents.*  
- **num_rel** *(int): Number of relevant documents.*  
- **num_rel_ret** *(int): Number of retrieved relevant documents.*  
- **num_q** *(int): Number of queries.*  
- **map** *(float): Mean average precision.*
- **gm_map** *(float): geometric mean average precision.*  
- **bpref** *(float): binary preference score.*  
- **Rprec** *(float): precision@R, where R is number of relevant documents.*  
- **recip_rank** *(float): reciprocal rank*  
- **P@k** *(float): precision@k (k in [5, 10, 15, 20, 30, 100, 200, 500, 1000]).*  
- **NDCG@k** *(float): nDCG@k (k in [5, 10, 15, 20, 30, 100, 200, 500, 1000]).*  

### Examples

A minimal example of looks as follows:
```Python
qrel = {
    "query": [0],
    "q0": ["q0"],
    "docid": ["doc_1"],
    "rel": [2]
}
run = {
    "query": [0, 0],
    "q0": ["q0", "q0"],
    "docid": ["doc_2", "doc_1"],
    "rank": [0, 1],
    "score": [1.5, 1.2],
    "system": ["test", "test"]
}

trec_eval = evaluate.load("trec_eval")
results = trec_eval.compute(references=[qrel], predictions=[run])
results["P@5"]
0.2
```

A more realistic use case with an examples from [`trectools`](https://github.com/joaopalotti/trectools):

```python
qrel = pd.read_csv("robust03_qrels.txt", sep="\s+", names=["query", "q0", "docid", "rel"])
qrel["q0"] = qrel["q0"].astype(str)
qrel = qrel.to_dict(orient="list")

run = pd.read_csv("input.InexpC2", sep="\s+", names=["query", "q0", "docid", "rank", "score", "system"])
run = run.to_dict(orient="list")

trec_eval = evaluate.load("trec_eval")
result = trec_eval.compute(run=[run], qrel=[qrel])
```

```python
result

{'runid': 'InexpC2',
 'num_ret': 100000,
 'num_rel': 6074,
 'num_rel_ret': 3198,
 'num_q': 100,
 'map': 0.22485930431817494,
 'gm_map': 0.10411523825735523,
 'bpref': 0.217511695914079,
 'Rprec': 0.2502547201167236,
 'recip_rank': 0.6646545943335417,
 'P@5': 0.44,
 'P@10': 0.37,
 'P@15': 0.34600000000000003,
 'P@20': 0.30999999999999994,
 'P@30': 0.2563333333333333,
 'P@100': 0.1428,
 'P@200': 0.09510000000000002,
 'P@500': 0.05242,
 'P@1000': 0.03198,
 'NDCG@5': 0.4101480395089769,
 'NDCG@10': 0.3806761417784469,
 'NDCG@15': 0.37819463408955706,
 'NDCG@20': 0.3686080836061317,
 'NDCG@30': 0.352474353427451,
 'NDCG@100': 0.3778329431025776,
 'NDCG@200': 0.4119129817248979,
 'NDCG@500': 0.4585354576461375,
 'NDCG@1000': 0.49092149290805653}
```

## Limitations and Bias
The `trec_eval` metric requires the inputs to be in the TREC run and qrel formats for predictions and references.


## Citation

```bibtex
@inproceedings{palotti2019,
 author = {Palotti, Joao and Scells, Harrisen and Zuccon, Guido},
 title = {TrecTools: an open-source Python library for Information Retrieval practitioners involved in TREC-like campaigns},
 series = {SIGIR'19},
 year = {2019},
 location = {Paris, France},
 publisher = {ACM}
} 
```

## Further References

- Homepage: https://github.com/joaopalotti/trectools
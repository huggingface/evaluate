---
title: G-Pass@k
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
    G-Pass@$k$ is a generalization of the Pass@$k$ metric, which evaluates both the stability and potential of large language models (LLMs) in reasoning tasks, described in https://arxiv.org/abs/2412.13147.
---

# Metric Card for GPassK

## Metric Description
G-Pass@$k$ is a generalization of the Pass@$k$ metric, which evaluates both the stability and potential of large language models (LLMs) in reasoning tasks. 

Given a threshold $\tau$, the G-Pass@$k_{\tau}$ measures the probability that a model will pass at least $m = \lceil \tau \cdot k \rceil$ out of $k$ attempts, 
where $c$ is the number of correct solutions and $n$ is the total number of generations.

$$
    \text{G-Pass@}k_{\tau} = \left[ \sum_{j = \lceil \tau \cdot k \rceil}^{c} \frac{\binom{c}{j} \cdot \binom{n - c}{k - j}}{\binom{n}{k}} \right]
$$

mG-Pass@$k$ extends the concept of G-Pass@$k_{\tau}$ by integrating over all thresholds from 0.5 to 1.0, 
effectively calculating the area under the curve of G-Pass@$k_{\tau}$. 
This provides an overall measure of how well the LLM performs across different levels of stringency.

$$
    \text{mG-Pass@}k = 2\int_{0.5}^{1.0} \text{G-Pass@}k_{\tau} d \tau = \frac{2}{k} \sum_{i= \lceil 0.5 \cdot k \rceil + 1}^{k} \text{G-Pass@}k_{\frac{i}{k}}
$$

## How to Use

### Inputs
- **predictions** (List[List[str]]): list of generations to evaluate. Each prediction should be a list of string with several model-generated solutions.
- **references** (List[str]): list of answer for each prediction.
- **k** (List[int]): list of number of attempts to consider in evaluation (Default: [4, 8, 16]).
- **thresholds** (List[float]): list of thresholds to consider in evaluation (Default: [0.25, 0.5, 0.75, 1.0]).
- **check_correct_fn** (Callable): function to check if a prediction is correct. It should have two parameters: `pred` and `ref` and output a boolean

### Output Values

The G-Pass@k metric returns one dict:
`g_pass_at_k`: dict with scores for each $k$ and threshold, and mG-Pass@$k$.

These metrics can take on any value between 0 and 1, inclusive. Higher scores are better.

#### Values from Popular Papers
The [leaderboard](https://open-compass.github.io/GPassK/) contains performance of several open-source and closed-source LLMs on the mathematical task.

### Examples
```python
from evaluate import load
g_pass_at_k_evaluator = evaluate.load("gpassk")
predictions = [["a", "b", "a", "a", "b", "a", "b", "c", "a", "c", "b", "a", "a", "b", "a", "b"]]
references = ["a"]
check_correct_fn = lambda pred, ref: pred == ref
g_pass_at_k = g_pass_at_k_evaluator.compute(predictions=predictions, 
    references=references, k=[4, 8], check_correct_fn=check_correct_fn)
print(g_pass_at_k)
{
    'G-Pass@4_0.25': 0.9615384615384616, 'G-Pass@4_0.5': 0.7153846153846154, 
    'G-Pass@4_0.75': 0.2846153846153846, 'G-Pass@4_1.0': 0.038461538461538464, 
    'G-Pass@8_0.25': 0.9949494949494949, 'G-Pass@8_0.5': 0.6903651903651904, 
    'G-Pass@8_0.75': 0.06596736596736597, 'G-Pass@8_1.0': 7.77000777000777e-05, 
    'mG-Pass@4': 0.16153846153846152, 'mG-Pass@8': 0.09518259518259518
}
```

## Citation
```bibtex
@misc{liu2024llmscapablestablereasoning,
      title={Are Your LLMs Capable of Stable Reasoning?}, 
      author={Junnan Liu and Hongwei Liu and Linchen Xiao and Ziyi Wang and Kuikun Liu and Songyang Gao and Wenwei Zhang and Songyang Zhang and Kai Chen},
      year={2024},
      eprint={2412.13147},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.13147}, 
}
```

## Further References

- [GPassK on github](https://github.com/open-compass/GPassK/)

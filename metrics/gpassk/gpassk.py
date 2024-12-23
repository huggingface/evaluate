# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of G-Pass@k Metirc Described in https://arxiv.org/abs/2412.13147."""
from typing import List, Callable
from functools import partial
import inspect

import datasets
import numpy as np
from scipy.stats import hypergeom

import evaluate


_CITATION = """\
@misc{liu2024llmscapablestablereasoning,
      title={Are Your LLMs Capable of Stable Reasoning?}, 
      author={Junnan Liu and Hongwei Liu and Linchen Xiao and Ziyi Wang and Kuikun Liu and Songyang Gao and Wenwei Zhang and Songyang Zhang and Kai Chen},
      year={2024},
      eprint={2412.13147},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.13147}, 
}
"""


_DESCRIPTION = """\
G-Pass@:math:`k` is a generalization of the Pass@:math:`k` metric, which evaluates both the stability and potential of large language models (LLMs) in reasoning tasks. 
Given a threshold :math:`\tau`, the G-Pass@:math:`k_{\tau}` measures the probability that a model will pass at least :math:`m = \lceil \tau \cdot k \rceil` out of :math:`k` attempts, 
where :math:`c` is the number of correct solutions and :math:`n` is the total number of generations.

.. math::
    \text{G-Pass@}k_{\tau} = \left[ \sum_{j = \lceil \tau \cdot k \rceil}^{c} \frac{\binom{c}{j} \cdot \binom{n - c}{k - j}}{\binom{n}{k}} \right]

mG-Pass@:math:`k` extends the concept of G-Pass@:math:`k_{\tau}` by integrating over all thresholds from 0.5 to 1.0, 
effectively calculating the area under the curve of G-Pass@:math:`k_{\tau}`. 
This provides an overall measure of how well the LLM performs across different levels of stringency.

.. math::
    \text{mG-Pass@}k = 2\int_{0.5}^{1.0} \text{G-Pass@}k_{\tau} d \tau = \frac{2}{k} \sum_{i= \lceil 0.5 \cdot k \rceil + 1}^{k} \text{G-Pass@}k_{\frac{i}{k}}

"""


_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of generations to evaluate. Each prediction should be a
    list of string with several model-generated solutions.
    references: list of answer for each prediction.
    k: list of number of attempts to consider in evaluation (Default: [4, 8, 16]).
    thresholds: list of thresholds to consider in evaluation (Default: [0.25, 0.5, 0.75, 1.0]).
    check_correct_fn: function to check if a prediction is correct. 
    It should have two parameters: `pred` and `ref` and output a boolean.
Returns:
    g_pass_at_k: dict with scores for each k and threshold, and mG-Pass@k.
Examples:
    >>> g_pass_at_k_evaluator = evaluate.load("gpassk")
    >>> predictions = [["a", "b", "a", "a", "b", "a", "b", "c", "a", "c", "b", "a", "a", "b", "a", "b"]]
    >>> references = ["a"]
    >>> check_correct_fn = lambda pred, ref: pred == ref
    >>> g_pass_at_k = g_pass_at_k_evaluator.compute(predictions=predictions, 
    references=references, k=[4, 8], check_correct_fn=check_correct_fn)
    >>> print(g_pass_at_k)
    {'G-Pass@4_0.25': 0.9615384615384616, 'G-Pass@4_0.5': 0.7153846153846154, 
    'G-Pass@4_0.75': 0.2846153846153846, 'G-Pass@4_1.0': 0.038461538461538464, 
    'G-Pass@8_0.25': 0.9949494949494949, 'G-Pass@8_0.5': 0.6903651903651904, 
    'G-Pass@8_0.75': 0.06596736596736597, 'G-Pass@8_1.0': 7.77000777000777e-05, 
    'mG-Pass@4': 0.16153846153846152, 'mG-Pass@8': 0.09518259518259518}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class GPassK(evaluate.Metric):

    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                'predictions': datasets.Value('int64'),
                'references': datasets.Value('int64'),
            }),
            homepage="https://open-compass.github.io/GPassK/",
            codebase_urls=["https://github.com/open-compass/GPassK"],
            reference_urls=["http://arxiv.org/abs/2412.13147"]
        )

    def _compute(self, 
                 predictions: List[List[str]], 
                 references: List[str], 
                 k=[4, 8, 16], 
                 thresholds=[0.25, 0.5, 0.75, 1.0], 
                 check_correct_fn: Callable = None):
        """Compute GPassK metric."""

        if check_correct_fn is None:
            raise ValueError('`check_correct_fn` is required for GPassK metric')
        
        sig = inspect.signature(check_correct_fn)
        if len(sig.parameters) != 2:
            raise ValueError(f'`check_correct_fn` should have exactly 2 parameters, got {len(sig.parameters)}')
        for name in sig.parameters:
            if name not in ['pred', 'ref']:
                raise ValueError(f'`check_correct_fn` should have only `pred` and `ref` as parameters, got {name}')
        
        n_list, c_list = [], []
        for preds, ref in zip(predictions, references):
            labels = list(map(partial(check_correct_fn, ref=ref), preds))
            n = len(preds)
            c = sum(labels)
            n_list.append(n)
            c_list.append(c)
        
        g_pass_at_k = {
            f"G-Pass@{k_i}_{t}": np.mean([compute_g_pass_at_k(n, c, k_i, t) for n, c in zip(n_list, c_list)]).item()
            for k_i in k
            for t in thresholds
        }
        g_pass_at_k.update({
            f"mG-Pass@{k_i}": np.mean([compute_mg_pass_at_k(n, c, k_i) for n, c in zip(n_list, c_list)]).item()
            for k_i in k
        })
        return g_pass_at_k
    

def _compute_g_pass_at_k(n, c, k, m):
    if m > min(c, k) or k > n or c < 0 or n <= 0 or m < 0:
        return 0.0
    return hypergeom.sf(m - 1, n, c, k)


def compute_g_pass_at_k(n, c, k, t):
    m = max(int(np.ceil(k * t)), 1)
    return _compute_g_pass_at_k(n, c, k, m)


def compute_mg_pass_at_k(n, c, k):
    l, r = int(np.ceil(k * 0.5)), k

    mg_pass_at_k = 0.0
    for i in range(l + 1, r + 1):
        mg_pass_at_k += _compute_g_pass_at_k(n, c, k, i)
    mg_pass_at_k = 2 * mg_pass_at_k / k

    return mg_pass_at_k

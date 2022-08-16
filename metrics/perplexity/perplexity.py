# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Perplexity Metric."""

import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import evaluate
from evaluate import logging


_CITATION = """\

"""

_DESCRIPTION = """
Perplexity (PPL) is one of the most common metrics for evaluating language models.
It is defined as the exponentiated average negative log-likelihood of a sequence.

For more information, see https://huggingface.co/docs/transformers/perplexity
"""

_KWARGS_DESCRIPTION = """
Perplexity can be calculated by passing in a set of logits, labels, and attention mask tensors to the `compute()` function,
or by passing in a `model_id` and a list of texts to `texts` in the `compute_perplexity_with_pretrained_model()` function,
which will load a pretrained model and run inference.
Args for `compute`:
    logits (`ndarray`): Tensor-like, of shape [batch size, sequence length, vocab size]
    labels (`ndarray`): Tensor-like, of shape [batch, sequence length]
    attention_mask (`ndarray`): Tensor-like, of shape [batch, sequence length]
Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Perplexity(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float"),
                    "references": datasets.Value("float"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    def _compute(self, predictions, references, attention_mask=None):
        """
        Computes perplexity according to a set of logits, labels, and an attention mask.
        Args:
            predictions (`ndarray`): Logits, tensor-like, of shape [batch size, sequence length, vocab size]
            references (`ndarray`): Labels, tensor-like, of shape [batch, sequence length]
            attention_mask (`ndarray`): Tensor-like, of shape [batch, sequence length]

        Returns:
            (`dict`): Dictionary containing perplexity for each example and mean perplexity.
        """
        logits = predictions[..., :-1, :]
        labels = references[..., 1:]
        attention_mask = attention_mask[..., 1:]

        ppls = torch.exp(
            (CrossEntropyLoss(reduction="none")(logits.transpose(1, 2), labels) * attention_mask).sum(1)
            / attention_mask.sum(1)
        ).tolist()
        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

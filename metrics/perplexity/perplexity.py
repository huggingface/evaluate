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
Args for `compute_perplexity_with_pretrained_model`:
    model_id (str): model used for calculating Perplexity
            NOTE: Perplexity can only be calculated for causal language models.
                    This includes models such as gpt2, causal variations of bert,
                    causal versions of t5, and more (the full list can be found
                    in the AutoModelForCausalLM documentation here:
                    https://huggingface.co/docs/transformers/master/en/model_doc/auto#transformers.AutoModelForCausalLM )
    texts (list of str): input data, each separate text snippet is one list entry.
    batch_size (int): the batch size to run texts through the model. Defaults to 16.
    add_start_token (bool): whether to add the start token to the texts,
        so the perplexity can include the probability of the first word. Defaults to True.
    device (str): device to run on, defaults to `cuda` when available
Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.
Examples:
    Example 1:
        >>> perplexity = evaluate.load("perplexity", module_type="metric")
        >>> input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]
        >>> results = perplexity.compute_perplexity_with_pretrained_model(model_id='gpt2',
        ...                              add_start_token=False,
        ...                              texts=input_texts) # doctest:+ELLIPSIS
        >>> print(list(results.keys()))
        ['perplexities', 'mean_perplexity']
        >>> print(round(results["mean_perplexity"], 2))
        78.22
        >>> print(round(results["perplexities"][0], 2))
        11.11

    Example 2:
        >>> from datasets import load_dataset
        >>> perplexity = evaluate.load("perplexity", module_type="metric")
        >>> input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:10] # doctest: +SKIP
        >>> input_texts = [s for s in input_texts if s!='']
        >>> results = perplexity.compute_perplexity_with_pretrained_model(model_id='gpt2',
        ...                              predictions=input_texts)
        >>> print(list(results.keys()))
        ['perplexities', 'mean_perplexity']
        >>> print(round(results["mean_perplexity"], 2)) # doctest: +SKIP
        60.35
        >>> print(round(results["perplexities"][0], 2)) # doctest: +SKIP
        81.12
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
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    @staticmethod
    def _compute_perplexity_for_model(texts, model_id, batch_size, add_start_token, device):

        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = model.config.max_length - 1
        else:
            max_tokenized_len = model.config.max_length

        encodings = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp2(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

    def _compute(self, logits=None, labels=None, attention_mask=None):
        """
        Computes perplexity according to a set of logits, labels, and an attention mask.
        Args:
            logits (`ndarray`): Tensor-like, of shape [batch size, sequence length, vocab size]
            labels (`ndarray`): Tensor-like, of shape [batch, sequence length]
            attention_mask (`ndarray`): Tensor-like, of shape [batch, sequence length]

        Returns:
            (`dict`): Dictionary containing perplexity for each example and mean perplexity.
        """
        ppls = torch.exp(
            (CrossEntropyLoss(reduction="none")(logits.transpose(1, 2), labels) * attention_mask).sum(1)
            / attention_mask.sum(1)
        ).tolist()
        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

    def compute_perplexity_with_pretrained_model(
        self, texts, model_id: str = None, batch_size: int = 16, add_start_token: bool = True, device=None
    ):
        """
        Compute batched perplexity with regard to some pretrained model loaded by name
        Args:
            texts (`list` of `str`): List of texts to calculate perplexity over.
            model_id (`str`): Name of pretrained model to load.
            batch_size (`int`): Batch size to run inference on pretrained model.
            add_start_token (`bool`): Whether to add the start token to the texts, so the perplexity can include the probability of the first word. Defaults to True.
            device (`str`): device to run on, defaults to `cuda` when available
        Returns:
           (`dict`): Dictionary containing perplexity for each example and mean perplexity.
        """
        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        ppls = self._compute_perplexity_for_model(
            texts, model_id=model_id, batch_size=batch_size, add_start_token=add_start_token, device=device
        )
        return ppls

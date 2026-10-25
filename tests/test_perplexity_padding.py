import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

import evaluate

from .utils import require_torch, require_transformers


TEXTS = ["alpha beta", "alpha beta gamma delta epsilon"]
VOCAB = {
    token: i
    for i, token in enumerate(["<pad>", "<bos>", "<eos>", "<unk>", "alpha", "beta", "gamma", "delta", "epsilon"])
}
RTOL = 1e-5
ATOL = 1e-6
SEED = 636
REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_tokenizer(padding_side):
    backend = Tokenizer(WordLevel(VOCAB, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        padding_side=padding_side,
        model_max_length=32,
    )


def _build_model(architecture):
    torch.manual_seed(SEED)
    common = dict(vocab_size=len(VOCAB), bos_token_id=1, eos_token_id=2, pad_token_id=0)
    if architecture == "gpt2":
        model = GPT2LMHeadModel(
            GPT2Config(
                **common,
                n_positions=32,
                n_ctx=32,
                n_embd=24,
                n_layer=2,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )
    else:
        model = LlamaForCausalLM(
            LlamaConfig(
                **common,
                hidden_size=24,
                intermediate_size=48,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=32,
                attention_dropout=0.0,
            )
        )
    return model.to("cpu").eval()


def _unpadded_reference(model, tokenizer, add_start_token):
    """Independent teacher-forced scores: one unpadded forward pass per text.

    Without BOS, score text tokens starting at the second token. With BOS, score
    every text token conditioned on the preceding BOS/text prefix.
    """
    references = []
    for text in TEXTS:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        sequence = ([tokenizer.bos_token_id] if add_start_token else []) + ids
        inputs = torch.tensor([sequence], device="cpu")
        with torch.no_grad():
            logits = model(inputs, attention_mask=torch.ones_like(inputs)).logits[0]
        target_ids = inputs[0, 1:]
        log_probs = logits[:-1].double().log_softmax(-1)
        nlls = -log_probs.gather(1, target_ids[:, None]).squeeze(1)
        references.append(nlls.mean().exp().item())
    return references


def _load_module(module_type):
    return evaluate.load(str(REPO_ROOT / f"{module_type}s" / "perplexity"), module_type=module_type)


def _compute(module, module_type, texts, model_id, batch_size, add_start_token):
    input_key = "predictions" if module_type == "metric" else "data"
    return module.compute(
        **{input_key: texts},
        model_id=model_id,
        batch_size=batch_size,
        add_start_token=add_start_token,
        device="cpu",
    )


@pytest.fixture
def perplexity_offline_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setattr(evaluate.logging, "_tqdm_active", False)


@pytest.fixture(scope="module")
def saved_checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-perplexity")
    checkpoints = {}
    for architecture in ("gpt2", "llama"):
        model = _build_model(architecture)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 16000
        for padding_side in ("left", "right"):
            path = root / f"{architecture}-{padding_side}"
            model.save_pretrained(path)
            _build_tokenizer(padding_side).save_pretrained(path)
            checkpoints[(architecture, padding_side)] = str(path)
    return checkpoints


@require_torch
@require_transformers
@pytest.mark.parametrize("architecture", ["gpt2", "llama"])
@pytest.mark.parametrize("module_type", ["metric", "measurement"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("add_start_token", [False, True])
def test_perplexity_padding_matches_unpadded_reference(
    saved_checkpoints, perplexity_offline_env, architecture, module_type, padding_side, add_start_token
):
    model_id = saved_checkpoints[(architecture, padding_side)]
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True).to("cpu").eval()
    reference = _unpadded_reference(model, tokenizer, add_start_token)

    module = _load_module(module_type)
    separately = [
        _compute(module, module_type, [text], model_id, batch_size=1, add_start_token=add_start_token)["perplexities"][
            0
        ]
        for text in TEXTS
    ]
    together = _compute(module, module_type, TEXTS, model_id, batch_size=2, add_start_token=add_start_token)
    batch_one = _compute(module, module_type, TEXTS, model_id, batch_size=1, add_start_token=add_start_token)

    np.testing.assert_allclose(separately, reference, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(together["perplexities"], reference, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(batch_one["perplexities"], reference, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(together["perplexities"][1], reference[1], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(together["mean_perplexity"], np.mean(together["perplexities"]), rtol=RTOL, atol=ATOL)

    reloaded = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    assert reloaded.padding_side == padding_side
    tokenizer_config = json.loads((Path(model_id) / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tokenizer_config.get("padding_side") == padding_side

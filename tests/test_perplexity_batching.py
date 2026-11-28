from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate


TEXTS = ["alpha beta gamma delta", "alpha beta", "gamma alpha beta", "beta alpha", "alpha beta " * 40]


@pytest.fixture(scope="module")
def tiny_gpt2():
    torch = pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocabulary = {"<pad>": 0, "<bos>": 1, "<unk>": 2, "alpha": 3, "beta": 4, "gamma": 5, "delta": 6}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

    def make_tokenizer(padding_side):
        return transformers.PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="<pad>", bos_token="<bos>", padding_side=padding_side
        )

    config = transformers.GPT2Config(
        vocab_size=len(vocabulary),
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
        n_positions=64,
        n_layer=1,
        n_head=2,
        n_embd=16,
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            model = transformers.GPT2LMHeadModel(config).eval()
        yield torch, model, make_tokenizer
    finally:
        torch.set_num_threads(previous_threads)


def _globally_padded_perplexity(torch, model, tokenizer, add_start_token, max_length):
    encoded = tokenizer(
        TEXTS,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_length - int(add_start_token),
        return_tensors="pt",
    )
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    if add_start_token:
        ids = torch.cat((torch.full_like(ids[:, :1], tokenizer.bos_token_id), ids), dim=1)
        mask = torch.cat((torch.ones_like(mask[:, :1]), mask), dim=1)
    with torch.no_grad():
        logits = model(ids, attention_mask=mask).logits
    losses = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
    mask = mask[:, 1:]
    return ((losses * mask).sum(1) / mask.sum(1)).exp().tolist()


@pytest.mark.parametrize(("module_dir", "input_key"), [("metrics", "predictions"), ("measurements", "data")])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("add_start_token", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_perplexity_batches_preserve_scores(
    tiny_gpt2, monkeypatch, module_dir, input_key, padding_side, add_start_token, batch_size
):
    torch, model, make_tokenizer = tiny_gpt2
    tokenizer = make_tokenizer(padding_side)
    max_length = 48
    expected = _globally_padded_perplexity(torch, model, tokenizer, add_start_token, max_length)
    tokenizer_batch_sizes = []

    class RecordingTokenizer:
        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def __call__(self, texts, **kwargs):
            tokenizer_batch_sizes.append(len(texts))
            return tokenizer(texts, **kwargs)

    repo = Path(__file__).resolve().parents[1]
    module = evaluate.load(str(repo / module_dir / "perplexity"), keep_in_memory=True)
    module_globals = module._compute.__globals__
    monkeypatch.setitem(module_globals, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=lambda _name: model))
    monkeypatch.setitem(
        module_globals,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda _name: RecordingTokenizer()),
    )
    actual = module.compute(
        model_id="local-test-gpt2",
        batch_size=batch_size,
        max_length=max_length,
        device="cpu",
        add_start_token=add_start_token,
        **{input_key: TEXTS},
    )
    assert actual["perplexities"] == pytest.approx(expected, rel=1e-6, abs=1e-5)
    assert actual["mean_perplexity"] == pytest.approx(sum(expected) / len(expected), rel=1e-6, abs=1e-5)
    if padding_side == "right":
        assert tokenizer_batch_sizes == [len(TEXTS[i : i + batch_size]) for i in range(0, len(TEXTS), batch_size)]
    else:
        assert tokenizer_batch_sizes == [len(TEXTS)]

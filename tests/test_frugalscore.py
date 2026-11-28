from pathlib import Path

import pytest

import evaluate


def test_frugalscore_matches_direct_prediction(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")

    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "alpha": 4, "beta": 5, "gamma": 6}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )
    config = transformers.BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_labels=1,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = transformers.BertForSequenceClassification(config).eval()
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", lambda _name: model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda _name: tokenizer)
    monkeypatch.chdir(tmp_path)

    predictions = ["alpha", "beta gamma alpha", "alpha beta"]
    references = ["beta", "gamma beta", "gamma"]
    inputs = tokenizer(predictions, references, max_length=6, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        expected = model(**inputs).logits.squeeze(-1).tolist()
    metric_path = Path(__file__).resolve().parents[1] / "metrics" / "frugalscore"
    metric = evaluate.load(str(metric_path), keep_in_memory=True)
    result = metric.compute(predictions=predictions, references=references, batch_size=2, max_length=6, device="cpu")
    assert result["scores"] == pytest.approx(expected)

from unittest.mock import MagicMock

from metrics.bertscore.bertscore import _apply_tokenizer_max_length, _resolve_tokenizer_max_length


def test_resolve_tokenizer_max_length_prefers_explicit_value():
    tokenizer = MagicMock(model_max_length=10**30)
    assert _resolve_tokenizer_max_length(tokenizer, max_length=128) == 128


def test_resolve_tokenizer_max_length_uses_model_input_sizes():
    tokenizer = MagicMock(model_max_length=10**30, max_model_input_sizes={"microsoft/deberta-xlarge-mnli": 512})
    assert _resolve_tokenizer_max_length(tokenizer) == 512


def test_resolve_tokenizer_max_length_keeps_reasonable_default():
    tokenizer = MagicMock(model_max_length=512, max_model_input_sizes={})
    assert _resolve_tokenizer_max_length(tokenizer) == 512


def test_apply_tokenizer_max_length_updates_tokenizer():
    tokenizer = MagicMock(model_max_length=10**30, max_model_input_sizes={"x": 256})
    scorer = MagicMock(_tokenizer=tokenizer)
    assert _apply_tokenizer_max_length(scorer) == 256
    assert tokenizer.model_max_length == 256

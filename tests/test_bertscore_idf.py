from pathlib import Path

import pytest

import evaluate


def test_bertscore_refreshes_idf_for_changed_references(tmp_path):
    pytest.importorskip("bert_score")
    torch = pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    vocabulary = {
        token: index for index, token in enumerate("[PAD] [UNK] [CLS] [SEP] cat sat dog bird fish swam".split())
    }
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])]
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=64,
    )
    tokenizer.save_pretrained(tmp_path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        config = transformers.BertConfig(
            vocab_size=len(vocabulary),
            num_hidden_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            intermediate_size=32,
        )
        transformers.BertModel(config).save_pretrained(tmp_path)

    metric_path = Path(__file__).resolve().parents[1] / "metrics" / "bertscore"
    reused = evaluate.load(str(metric_path), keep_in_memory=True)
    fresh = evaluate.load(str(metric_path), keep_in_memory=True)
    options = {
        "predictions": ["cat sat", "fish swam"],
        "model_type": str(tmp_path),
        "num_layers": 1,
        "idf": True,
        "nthreads": 0,
        "device": "cpu",
        "use_fast_tokenizer": True,
    }
    reused.compute(references=["cat dog", "cat bird"], **options)
    cached_scorer = reused.cached_bertscorer

    references = [["cat sat"], ["fish cat", "bird fish"]]
    expected = fresh.compute(references=references, **options)
    assert reused.compute(references=references, **options) == expected
    assert reused.cached_bertscorer is cached_scorer

import importlib

import pytest

import evaluate


def require_pycoco():
    return importlib.util.find_spec("pycocoevalcap") is not None


@pytest.mark.skipif(not require_pycoco(), reason="pycocoevalcap not installed")
def test_bleu_coco_tokenizer_matches_reported_example():
    bleu = evaluate.load("bleu")

    preds = ["opacity, consolidation, pleural effusion, and atelectasis are present."]
    refs = ["opacity, consolidation, pleural effusion, and pneumonia are present."]

    # evaluate with COCO/PTB tokenization
    res_coco = bleu.compute(predictions=preds, references=refs, tokenizer_name="coco")
    # evaluate default tokenizer to ensure different score
    res_default = bleu.compute(predictions=preds, references=refs)

    assert res_coco["bleu"] != pytest.approx(res_default["bleu"])  # ensure difference exists
    # Expected around 0.5946035573 vs ~0.70 for default/period example
    assert res_coco["bleu"] == pytest.approx(0.5946035573, rel=1e-6, abs=1e-6)


@pytest.mark.skipif(not require_pycoco(), reason="pycocoevalcap not installed")
def test_bleu_coco_tokenizer_period_case():
    bleu = evaluate.load("bleu")

    preds = ["opacity . consolidation . pleural effusion . atelectasis are present ."]
    refs = ["opacity . consolidation . pleural effusion . pneumonia are present ."]

    res_coco = bleu.compute(predictions=preds, references=refs, tokenizer_name="coco")
    assert res_coco["bleu"] == pytest.approx(0.7016879389890388, rel=1e-6, abs=1e-6)



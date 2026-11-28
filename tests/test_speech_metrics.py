from pathlib import Path

import pytest

import evaluate


@pytest.mark.parametrize("metric_name", ["wer", "cer"])
@pytest.mark.parametrize(
    ("references", "predictions", "expected_wer", "expected_cer"),
    [
        (["", ""], ["", ""], 0, 0),
        (["", ""], ["alpha beta", "a"], 3, 11),
        (["   ", ""], ["alpha beta", "a"], 3, 11),
        (["hello world", ""], ["hello there", "x"], 1, 6 / 11),
    ],
)
def test_speech_metric_with_empty_references(metric_name, references, predictions, expected_wer, expected_cer):
    jiwer = pytest.importorskip("jiwer")
    if hasattr(jiwer, "compute_measures"):
        pytest.skip("Empty references require JiWER 4 or later")

    metric_path = Path(__file__).resolve().parents[1] / "metrics" / metric_name
    metric = evaluate.load(str(metric_path), keep_in_memory=True)
    expected = {"wer": expected_wer, "cer": expected_cer}[metric_name]
    assert metric.compute(references=references, predictions=predictions) == expected

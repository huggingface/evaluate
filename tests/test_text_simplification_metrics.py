from pathlib import Path

import pytest

import evaluate


@pytest.mark.parametrize("name", ["sari", "wiki_split"])
def test_text_simplification_scores_with_multiple_references(name):
    pytest.importorskip("sacrebleu")
    pytest.importorskip("sacremoses")

    metric_path = Path(__file__).resolve().parents[1] / "metrics" / name
    metric = evaluate.load(str(metric_path), keep_in_memory=True)
    result = metric.compute(
        sources=["The puppy was very small.", "The path was straight and narrow."],
        predictions=["The puppy was tiny.", "The path was straight."],
        references=[
            ["The puppy was tiny.", "The puppy was small."],
            ["The path was straight.", "The trail was straight and narrow."],
        ],
    )

    expected = {"sari": 68.87654420742656}
    if name == "wiki_split":
        expected.update(sacrebleu=100.0, exact=100.0)
    assert result == pytest.approx(expected)

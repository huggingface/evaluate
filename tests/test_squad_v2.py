from pathlib import Path

import pytest

import evaluate


SQUAD_V2_PATH = Path(__file__).resolve().parents[1] / "metrics" / "squad_v2"


@pytest.mark.parametrize(
    "prediction, answers, expected_exact, expected_f1",
    [
        ("the co-op co-op alpha", ["coop beta", "a co-op co-op", "THE!"], 0.0, 80.0),
        (" The co-op's plan.\t", ["a plan", "COOPS plan"], 100.0, 100.0),
        ("the !", [], 100.0, 100.0),
        ("something", ["The!", ""], 0.0, 0.0),
    ],
)
def test_squad_v2_scores(prediction, answers, expected_exact, expected_f1, tmp_path):
    metric = evaluate.load(str(SQUAD_V2_PATH), cache_dir=str(tmp_path), keep_in_memory=True)
    result = metric.compute(
        predictions=[{"id": "question", "prediction_text": prediction, "no_answer_probability": 0.0}],
        references=[{"id": "question", "answers": {"text": answers, "answer_start": [0] * len(answers)}}],
    )

    assert result["exact"] == expected_exact
    assert result["f1"] == pytest.approx(expected_f1)
    assert result["total"] == 1

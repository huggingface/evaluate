from pathlib import Path

import numpy as np
import pytest

import evaluate


@pytest.fixture
def brier_score():
    return evaluate.load(str(Path(__file__).resolve().parents[1] / "metrics" / "brier_score"), keep_in_memory=True)


@pytest.mark.parametrize("labels", [(0, 1), (-1, 1), (0, 2), (1, 2), (-2, -1), (2, 3)])
@pytest.mark.parametrize("sample_weight", [None, [1, 2, 3, 4]])
def test_brier_score_infers_positive_label(brier_score, labels, sample_weight):
    references = [labels[0], labels[0], labels[1], labels[1]]
    predictions = [0.1, 0.2, 0.8, 0.9]
    expected = np.average((np.array(predictions) - [0, 0, 1, 1]) ** 2, weights=sample_weight)

    result = brier_score.compute(references=references, predictions=predictions, sample_weight=sample_weight)

    assert result["brier_score"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "labels, pos_label", [((0, 2), 0), ((1, 2), 1), (("ham", "spam"), "ham"), (("ham", "spam"), "spam")]
)
def test_brier_score_respects_explicit_positive_label(brier_score, labels, pos_label):
    references = [labels[0], labels[0], labels[1], labels[1]]
    predictions = [0.1, 0.2, 0.8, 0.9]
    sample_weight = [1, 2, 3, 4]
    targets = np.array([label == pos_label for label in references])
    expected = np.average((np.array(predictions) - targets) ** 2, weights=sample_weight)

    result = brier_score.compute(
        references=references, predictions=predictions, sample_weight=sample_weight, pos_label=pos_label
    )

    assert result["brier_score"] == pytest.approx(expected)


def test_brier_score_requires_explicit_positive_label_for_strings(brier_score):
    with pytest.raises(ValueError, match="pos_label"):
        brier_score.compute(references=["ham", "spam"], predictions=[0.2, 0.8])

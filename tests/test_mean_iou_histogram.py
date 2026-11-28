from pathlib import Path

import numpy as np
import pytest

import evaluate


@pytest.mark.parametrize(
    ("predictions", "references", "num_labels", "iou"),
    [
        ([[0, 1, 2, 255], [2, 1, 0, 255]], [[0, 1, 1, 255], [2, 2, 0, 2]], 3, [1.0, 1 / 3, 1 / 4]),
        ([[0, 1, 255]], [[0, 0, 255]], 1, [0.5]),
    ],
)
def test_mean_iou_counts_integer_labels(predictions, references, num_labels, iou):
    pytest.importorskip("PIL")
    path = Path(__file__).resolve().parents[1] / "metrics" / "mean_iou"
    metric = evaluate.load(str(path), keep_in_memory=True)
    result = metric.compute(
        predictions=[np.array(predictions, dtype=np.uint8)],
        references=[np.array(references, dtype=np.uint8)],
        num_labels=num_labels,
        ignore_index=255,
    )
    assert result["per_category_iou"] == pytest.approx(iou)

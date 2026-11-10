from pathlib import Path

import numpy as np
import pytest

import evaluate


MASE_PATH = str(Path(__file__).resolve().parents[1] / "metrics" / "mase")


@pytest.mark.parametrize(
    "multioutput, expected",
    [("uniform_average", 0.75), ([0.25, 0.75], 0.625), ("raw_values", [1.0, 0.5])],
    ids=["uniform", "weighted", "raw"],
)
def test_mase_aggregates_scaled_errors(multioutput, expected):
    metric = evaluate.load(MASE_PATH, "multilist", keep_in_memory=True)

    for scale in (1, 10):
        result = metric.compute(
            predictions=[[1, 50 * scale], [1, 50 * scale]],
            references=[[0, 0], [0, 0]],
            training=[[0, 0], [1, 100 * scale], [2, 200 * scale]],
            multioutput=multioutput,
        )
        np.testing.assert_allclose(result["mase"], expected)


def test_mase_univariate_with_sample_weights_and_seasonality():
    metric = evaluate.load(MASE_PATH, keep_in_memory=True)

    result = metric.compute(
        predictions=[0, 4], references=[0, 0], training=[0, 100, 2, 104], periodicity=2, sample_weight=[1, 3]
    )

    assert result["mase"] == pytest.approx(1.0)


def test_mase_rejects_mismatched_training_outputs():
    metric = evaluate.load(MASE_PATH, "multilist", keep_in_memory=True)

    with pytest.raises(ValueError, match="same number of outputs"):
        metric.compute(predictions=[[1, 2]], references=[[0, 0]], training=[[0], [1], [2]])


@pytest.mark.parametrize("multioutput", ["variance_weighted", [1.0]])
def test_mase_rejects_invalid_output_aggregation(multioutput):
    metric = evaluate.load(MASE_PATH, "multilist", keep_in_memory=True)

    with pytest.raises(ValueError):
        metric.compute(predictions=[[1, 2]], references=[[0, 0]], training=[[0, 0], [1, 1]], multioutput=multioutput)

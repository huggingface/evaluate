import tracemalloc
from pathlib import Path

import numpy as np
import pytest

import evaluate


@pytest.fixture
def mahalanobis():
    metric_dir = Path(__file__).resolve().parents[1] / "metrics" / "mahalanobis"
    return evaluate.load(str(metric_dir), keep_in_memory=True)


def test_mahalanobis_scores_each_observation(mahalanobis):
    result = mahalanobis.compute(
        X=[[1, 2], [-3, 1], [0, 0]],
        reference_distribution=[[-2, -1], [2, 1], [1, 3], [-1, -3]],
    )

    np.testing.assert_allclose(result["mahalanobis"], [0.6, 7.5, 0.0])


def test_mahalanobis_does_not_allocate_pairwise_matrix(mahalanobis):
    num_observations = 4096
    observations = np.tile([2, 1], (num_observations, 1))
    tracemalloc.start()
    try:
        result = mahalanobis.compute(
            X=observations,
            reference_distribution=[[0, 0], [0, 2], [2, 0], [2, 2]],
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    np.testing.assert_allclose(result["mahalanobis"], np.full(num_observations, 0.75))
    # The old float64 pairwise matrix alone uses 128 MiB for this input.
    assert peak_bytes < 32 * 1024**2

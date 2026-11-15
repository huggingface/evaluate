from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset
from scipy.stats import bootstrap

import evaluate


class LocalClassificationPipeline:
    task = "text-classification"

    def __call__(self, inputs, **kwargs):
        return [{"label": int(row["text"])} for row in inputs]


@pytest.mark.parametrize("metric_name,options", [("accuracy", {"normalize": False}), ("f1", {"average": "macro"})])
def test_bootstrap_preserves_metric_options(metric_name, options):
    data = Dataset.from_dict(
        {
            "text": ["0", "1", "2", "0", "1", "2", "0", "1", "2", "0"],
            "label": [0, 1, 2, 1, 2, 0, 0, 1, 2, 2],
        }
    )
    metric = evaluate.load(str(Path(__file__).parents[1] / "metrics" / metric_name))
    task = evaluate.evaluator("text-classification")
    task.METRIC_KWARGS = options
    inputs = {"model_or_pipeline": LocalClassificationPipeline(), "data": data, "metric": metric, "device": -1}
    score = task.compute(**inputs)[metric_name]

    with mock.patch.object(metric, "_compute", wraps=metric._compute) as compute:
        result = task.compute(**inputs, strategy="bootstrap", n_resamples=19, random_state=0)[metric_name]
    assert result["score"] == score
    assert compute.call_count > 1
    for call in compute.call_args_list:
        for key, value in options.items():
            assert call.kwargs.get(key) == value
    if metric_name == "accuracy":
        assert result["confidence_interval"][1] > 1


@pytest.mark.parametrize("normalize", [True, False])
def test_bootstrap_resamples_sample_weights(normalize):
    predictions = np.arange(10)
    references = np.array([0, 1, 2, 3, 4, 5, 10, 10, 10, 10])
    weights = list(range(1, 11))
    data = Dataset.from_dict({"text": [str(value) for value in predictions], "label": references})
    metric = evaluate.load(str(Path(__file__).parents[1] / "metrics" / "accuracy"))
    task = evaluate.evaluator("text-classification")
    task.METRIC_KWARGS = {"sample_weight": weights, "normalize": normalize}

    def weighted_accuracy(predictions, references, sample_weight):
        correct = predictions == references
        return float(np.average(correct, weights=sample_weight) if normalize else np.dot(correct, sample_weight))

    expected = bootstrap(
        (predictions, references, np.array(weights)),
        weighted_accuracy,
        paired=True,
        vectorized=False,
        n_resamples=99,
        random_state=0,
    )
    with mock.patch.object(metric, "_compute", wraps=metric._compute) as compute:
        result = task.compute(
            model_or_pipeline=LocalClassificationPipeline(),
            data=data,
            metric=metric,
            device=-1,
            strategy="bootstrap",
            n_resamples=99,
            random_state=0,
        )["accuracy"]
    assert result["score"] == pytest.approx(weighted_accuracy(predictions, references, weights))
    assert result["confidence_interval"] == pytest.approx(tuple(expected.confidence_interval))
    assert result["standard_error"] == pytest.approx(expected.standard_error)
    for call in compute.call_args_list:
        np.testing.assert_array_equal(call.kwargs["sample_weight"], np.asarray(call.kwargs["predictions"]) + 1)
    assert task.METRIC_KWARGS == {"sample_weight": weights, "normalize": normalize}

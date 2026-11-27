import pytest
from scipy.stats import bootstrap

import evaluate
from evaluate.evaluator.base import Evaluator


class _DummyEvaluator(Evaluator):
    def __init__(self, metric_kwargs=None):
        self.task = "text-classification"
        self.default_metric_name = "accuracy"
        self.METRIC_KWARGS = dict(metric_kwargs or {})

    def predictions_processor(self, *args, **kwargs):
        return {}


def test_bootstrap_forwards_fixed_metric_kwargs_accuracy():
    metric = evaluate.load("./metrics/accuracy")
    preds = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    refs = [0, 1, 2, 1, 2, 0, 0, 1, 2, 2]
    ev = _DummyEvaluator({"normalize": False})

    result = ev.compute_metric(
        metric,
        {"predictions": preds, "references": refs},
        strategy="bootstrap",
        n_resamples=49,
        random_state=0,
    )

    acc = result["accuracy"]
    assert acc["score"] == pytest.approx(6.0)
    low, high = acc["confidence_interval"]
    assert low >= 1.0
    assert high <= 10.0
    assert low <= acc["score"] <= high


def test_bootstrap_forwards_fixed_metric_kwargs_multiclass_f1():
    metric = evaluate.load("./metrics/f1")
    preds = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    refs = [0, 1, 2, 1, 2, 0, 0, 1, 2, 2]
    ev = _DummyEvaluator({"average": "macro"})

    result = ev.compute_metric(
        metric,
        {"predictions": preds, "references": refs},
        strategy="bootstrap",
        n_resamples=49,
        random_state=0,
    )

    f1 = result["f1"]
    assert 0.0 <= f1["score"] <= 1.0
    low, high = f1["confidence_interval"]
    assert 0.0 <= low <= f1["score"] <= high <= 1.0


def test_bootstrap_resamples_sample_weight_and_does_not_mutate_metric_kwargs():
    metric = evaluate.load("./metrics/accuracy")
    preds = [0, 1, 0, 1, 0, 1, 0, 1]
    refs = [0, 1, 1, 1, 0, 0, 0, 1]
    weights = [3.0, 1.0, 2.0, 1.0, 4.0, 1.0, 2.0, 1.0]
    original_kwargs = {"normalize": False, "sample_weight": weights}
    ev = _DummyEvaluator(original_kwargs)

    result = ev.compute_metric(
        metric,
        {"predictions": preds, "references": refs},
        strategy="bootstrap",
        n_resamples=49,
        random_state=0,
    )

    assert ev.METRIC_KWARGS == original_kwargs

    def expected_stat(p, r, w):
        return metric.compute(predictions=p, references=r, sample_weight=w, normalize=False)["accuracy"]

    expected_bs = bootstrap(
        data=[preds, refs, weights],
        statistic=expected_stat,
        paired=True,
        vectorized=False,
        n_resamples=49,
        random_state=0,
    )
    acc = result["accuracy"]
    assert acc["score"] == pytest.approx(12.0)
    assert acc["confidence_interval"][0] == pytest.approx(expected_bs.confidence_interval.low)
    assert acc["confidence_interval"][1] == pytest.approx(expected_bs.confidence_interval.high)
    assert acc["standard_error"] == pytest.approx(expected_bs.standard_error)


def test_compute_confidence_interval_backwards_compatible_without_metric_kwargs():
    metric = evaluate.load("./metrics/accuracy")
    res = Evaluator._compute_confidence_interval(
        metric,
        {"predictions": [0, 1, 0, 1], "references": [0, 1, 1, 1]},
        ["accuracy"],
        n_resamples=19,
        random_state=0,
    )
    assert "accuracy" in res
    assert 0.0 <= res["accuracy"]["confidence_interval"][0] <= res["accuracy"]["confidence_interval"][1] <= 1.0

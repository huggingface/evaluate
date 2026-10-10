from unittest import TestCase, mock

import pytest
from datasets import Dataset

from evaluate import EvaluationSuite
from evaluate.evaluation_suite import SubTask
from tests.test_evaluator import DummyTextClassificationPipeline
from tests.test_metric import DummyMetric


class TestEvaluationSuite(TestCase):
    def setUp(self):
        # Check that the EvaluationSuite loads successfully
        self.evaluation_suite = EvaluationSuite.load("evaluate/evaluation-suite-ci")

        # Setup a dummy model for usage with the EvaluationSuite
        self.dummy_model = DummyTextClassificationPipeline()

    def test_running_evaluation_suite(self):

        # Check that the evaluation suite successfully runs
        results = self.evaluation_suite.run(self.dummy_model)

        # Check that the results are correct
        for r in results:
            self.assertEqual(r["accuracy"], 0.5)

        # Check that correct number of tasks were run
        self.assertEqual(len(results), 2)

    def test_empty_suite(self):

        self.empty_suite = self.evaluation_suite
        self.empty_suite.suite = []
        self.assertRaises(ValueError, self.empty_suite.run, self.dummy_model)


@pytest.mark.parametrize("configuration", ["none", "empty", "metric"])
@pytest.mark.parametrize("raises", [False, True])
def test_run_preserves_task_arguments(configuration, raises, tmp_path):
    data = Dataset.from_dict({"text": ["positive", "negative"], "label": [1, 0]})
    metric = DummyMetric(cache_dir=str(tmp_path))
    args_for_task = {"metric": metric} if configuration == "metric" else {} if configuration == "empty" else None
    original_args = args_for_task.copy() if args_for_task is not None else None
    task = SubTask("text-classification", data=data, args_for_task=args_for_task)
    suite = EvaluationSuite("local")
    suite.suite = [task]

    class LocalPipeline:
        task = "text-classification"

        def __call__(self, inputs, **kwargs):
            if raises:
                raise RuntimeError("pipeline failed")
            return [{"label": int(row["text"] == "positive")} for row in inputs]

    with mock.patch("evaluate.evaluator.base.load", return_value=metric) as load_metric:
        if raises:
            with pytest.raises(RuntimeError, match="pipeline failed"):
                suite.run(LocalPipeline())
        else:
            results = suite.run(LocalPipeline())
            assert results[0]["accuracy"] == 1.0
            assert results[0]["task_name"] is data

        if configuration == "metric":
            load_metric.assert_not_called()
        else:
            load_metric.assert_called_once_with("accuracy")

    assert task.args_for_task is args_for_task
    assert args_for_task == original_args


def test_run_preserves_shared_arguments_between_tasks(tmp_path):
    metric = DummyMetric(cache_dir=str(tmp_path))
    args_for_task = {"metric": metric, "label_mapping": {"POSITIVE": 1, "NEGATIVE": 0}}
    original_args = args_for_task.copy()
    first_data = Dataset.from_dict({"text": ["a", "b"], "label": [1, 0]})
    second_data = Dataset.from_dict({"text": ["c", "d"], "label": [0, 1]})
    suite = EvaluationSuite("local")
    suite.suite = [
        SubTask("text-classification", data=first_data, args_for_task=args_for_task),
        SubTask("text-classification", data=second_data, args_for_task=args_for_task),
    ]

    for _ in range(2):
        results = suite.run(DummyTextClassificationPipeline())
        assert [result["accuracy"] for result in results] == [1.0, 0.0]
        assert results[0]["task_name"] is first_data
        assert results[1]["task_name"] is second_data
        assert all(task.args_for_task is args_for_task for task in suite.suite)
        assert args_for_task == original_args

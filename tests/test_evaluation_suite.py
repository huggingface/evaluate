import os
import tempfile
from pathlib import Path
from unittest import TestCase

from datasets import Dataset

from evaluate import EvaluationSuite, load
from evaluate.evaluation_suite import SubTask
from tests.test_evaluator import DummyTextClassificationPipeline


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


class _LabelFromTextClassifier:
    task = "text-classification"

    def __init__(self, expected_text="1 processed", should_fail=False):
        self.expected_text = expected_text
        self.should_fail = should_fail

    def __call__(self, rows, **kwargs):
        if self.should_fail:
            raise RuntimeError("pipeline failure")
        return [{"label": int(row["text"] == self.expected_text)} for row in rows]


class TestEvaluationSuitePreprocessing(TestCase):
    def _accuracy(self):
        return load(os.path.join("metrics", "accuracy"), keep_in_memory=True)

    def test_in_memory_dataset_preprocessing_and_reusable_runs(self):
        data = Dataset.from_dict({"text": ["0", "1"], "label": [0, 1]})
        task = SubTask(
            "text-classification",
            data=data,
            data_preprocessor=lambda row: {"text": row["text"] + " processed"},
            args_for_task={"metric": self._accuracy(), "device": -1},
        )
        suite = EvaluationSuite("local-in-memory")
        suite.suite = [task]

        first = suite.run(_LabelFromTextClassifier())
        second = suite.run(_LabelFromTextClassifier())

        self.assertEqual(first[0]["accuracy"], 1.0)
        self.assertEqual(second[0]["accuracy"], 1.0)
        self.assertIs(task.data, data)
        self.assertEqual(task.data["text"], ["0", "1"])

    def test_path_backed_preprocessing_preserves_source_across_retries_and_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "json_data"
            data_dir.mkdir()
            (data_dir / "test.jsonl").write_text(
                '{"text": "0", "label": 0}\n{"text": "1", "label": 1}\n', encoding="utf-8"
            )

            task = SubTask(
                "text-classification",
                data=str(data_dir),
                split="test",
                data_preprocessor=lambda row: {"text": row["text"] + " processed"},
                args_for_task={"metric": self._accuracy(), "device": -1},
            )
            suite = EvaluationSuite("local-json")
            suite.suite = [task]

            with self.assertRaises(RuntimeError):
                suite.run(_LabelFromTextClassifier(should_fail=True))

            self.assertEqual(task.data, str(data_dir))

            first = suite.run(_LabelFromTextClassifier())
            second = suite.run(_LabelFromTextClassifier())

            self.assertEqual(first[0]["accuracy"], 1.0)
            self.assertEqual(second[0]["accuracy"], 1.0)
            self.assertEqual(task.data, str(data_dir))
            self.assertEqual(first[0]["task_name"], str(data_dir))

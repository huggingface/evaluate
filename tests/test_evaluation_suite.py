from unittest import TestCase

from evaluate import EvaluationSuite
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

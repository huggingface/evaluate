from unittest import TestCase

from evaluate import EvaluationSuite


class TestEvaluationSuite(TestCase):
    def setUp(self):
        self.evaluation_suite = EvaluationSuite.load("evaluate/evaluation-suite-ci")

    def test_running_evaluation_suite(self):
        results = self.evaluation_suite.run("lvwerra/distilbert-imdb")

        # Check that the results are correct and task ids are generated as expected
        self.assertEqual(results["task_d868de929b011184"]["result"]["accuracy"], 1.0)

        # Check that correct number of tasks were run
        self.assertEqual(len(list(results.keys())), 2)

from unittest import TestCase

from evaluate import EvaluationSuite


class TestEvaluationSuite(TestCase):

    def setUp(self):
        # Check that the EvaluationSuite loads successfully
        self.evaluation_suite = EvaluationSuite.load("evaluate/evaluation-suite-ci")

    def test_running_evaluation_suite(self):

        # Check that the evaluation suite successfully runs
        results = self.evaluation_suite.run("lvwerra/distilbert-imdb")

        # Check that the results are correct
        for r in results:
            self.assertEqual(r["accuracy"], 1.0)

        # Check that correct number of tasks were run
        self.assertEqual(len(results), 2)

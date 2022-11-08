from unittest import TestCase

from evaluate import EvaluationSuite


class TestEvaluationSuite(TestCase):
    def test_suite(self):
        suite = EvaluationSuite.load("evaluate/evaluation-suite-ci")
        results = suite.run("philschmid/tiny-bert-sst2-distilled")
        self.assertEqual(results["imdb"]["accuracy"], 1.0)

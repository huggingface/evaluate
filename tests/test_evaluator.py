from unittest import TestCase

from datasets import load_dataset, load_metric
from transformers import pipeline

from evaluate import Evaluator


class TestEvaluator(TestCase):
    def test_evaluator(self):
        pipe = pipeline("text-classification")
        metric = load_metric("accuracy")

        ds = load_dataset("imdb")
        ds = ds["test"].shuffle().select(range(32))  # just for speed
        ds = ds.rename_columns({"text": "inputs", "label": "references"})

        evaluator = Evaluator(pipe, ds, metric, label_mapping={"NEGATIVE": 0, "POSITIVE": 1})
        print(evaluator.compute())

        evaluator = Evaluator("text-classification", ds, "accuracy", label_mapping={"NEGATIVE": 0, "POSITIVE": 1})
        print(evaluator.compute())

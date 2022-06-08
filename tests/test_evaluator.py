# Copyright 2022 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

from unittest import TestCase

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from evaluate import TextClassificationEvaluator, evaluator, load


class TestEvaluator(TestCase):
    def setUp(self):
        self.data = load_dataset("imdb", split="test[:2]")
        self.input_column = "text"
        self.label_column = "label"
        self.pipe = pipeline("text-classification")
        self.evaluator = evaluator("text-classification")
        self.label_mapping = {"NEGATIVE": 0.0, "POSITIVE": 1.0}

    def test_pipe_init(self):
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            input_column="text",
            label_column="label",
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 1.0})

    def test_model_init(self):
        scores = self.evaluator.compute(
            model_or_pipeline="huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli",
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
        )
        self.assertEqual(scores, {"accuracy": 0.5})
        model = AutoModelForSequenceClassification.from_pretrained(
            "huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli"
        )
        tokenizer = AutoTokenizer.from_pretrained("huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli")
        scores = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="accuracy",
            tokenizer=tokenizer,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
        )
        self.assertEqual(scores, {"accuracy": 0.5})

    def test_class_init(self):
        evaluator = TextClassificationEvaluator()
        self.assertEqual(evaluator.task, "text-classification")
        self.assertIsNone(evaluator.default_metric_name)

        scores = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="f1",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"f1": 0.0})

    def test_default_pipe_init(self):
        scores = self.evaluator.compute(
            data=self.data,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 1.0})

    def test_overwrite_default_metric(self):
        accuracy = load("accuracy")
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=accuracy,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 1.0})
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 1.0})

    def test_bootstrap(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            "huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli"
        )
        tokenizer = AutoTokenizer.from_pretrained("huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli")
        results = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="accuracy",
            tokenizer=tokenizer,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
            strategy="bootstrap",
            n_resamples=10,
            random_state=0,
        )
        self.assertEqual(results["accuracy"]["score"], 0.5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][0], 0.0, 5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][1], 0.5, 5)
        self.assertAlmostEqual(results["accuracy"]["standard_error"], 0.3689323936863109, 5)

    def test_wrong_task(self):
        self.assertRaises(KeyError, evaluator, "bad_task")

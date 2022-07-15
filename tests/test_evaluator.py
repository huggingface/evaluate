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

from datasets import Dataset
from PIL import Image
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from evaluate import ImageClassificationEvaluator, TextClassificationEvaluator, evaluator, load


class DummyTextClassificationPipeline:
    def __init__(self):
        self.task = "text-classification"

    def __call__(self, text, **kwargs):
        return [{"label": "NEGATIVE"} if i % 2 == 1 else {"label": "POSITIVE"} for i, _ in enumerate(text)]


class DummyImageClassificationPipeline:
    def __init__(self):
        self.task = "image-classification"

    def __call__(self, images, **kwargs):
        return [[{"score": 0.9, "label": "yurt"}, {"score": 0.1, "label": "umbrella"}] for i, _ in enumerate(images)]


class TestEvaluator(TestCase):
    def test_wrong_task(self):
        self.assertRaises(KeyError, evaluator, "bad_task")


class TestTextClassificationEvaluator(TestCase):
    def setUp(self):
        self.data = Dataset.from_dict({"label": [1, 0], "text": ["great movie", "horrible movie"]})
        self.default_model = "lvwerra/distilbert-imdb"
        self.input_column = "text"
        self.label_column = "label"
        self.pipe = DummyTextClassificationPipeline()
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
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 1.0})
        model = AutoModelForSequenceClassification.from_pretrained(self.default_model)
        tokenizer = AutoTokenizer.from_pretrained(self.default_model)
        scores = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="accuracy",
            tokenizer=tokenizer,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 1.0})

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
        self.assertEqual(scores, {"f1": 1.0})

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
        data = Dataset.from_dict({"label": [1, 0, 0], "text": ["great movie", "great movie", "horrible movie"]})
        model = AutoModelForSequenceClassification.from_pretrained(self.default_model)
        tokenizer = AutoTokenizer.from_pretrained(self.default_model)

        results = self.evaluator.compute(
            model_or_pipeline=model,
            data=data,
            metric="accuracy",
            tokenizer=tokenizer,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
            strategy="bootstrap",
            n_resamples=10,
            random_state=0,
        )
        self.assertAlmostEqual(results["accuracy"]["score"], 0.666666, 5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][0], 0.33333, 5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][1], 0.68326, 5)
        self.assertAlmostEqual(results["accuracy"]["standard_error"], 0.24595, 5)


class TestImageClassificationEvaluator(TestCase):
    def setUp(self):
        self.data = Dataset.from_dict(
            {
                "labels": [2, 2],
                "image": [Image.new("RGB", (500, 500), (255, 255, 255)), Image.new("RGB", (500, 500), (170, 95, 170))],
            }
        )
        self.default_model = "lysandre/tiny-vit-random"
        self.input_column = "image"
        self.label_column = "labels"
        self.pipe = DummyImageClassificationPipeline()
        self.evaluator = evaluator("image-classification")
        self.label_mapping = AutoConfig.from_pretrained(self.default_model).label2id

    def test_pipe_init(self):
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            input_column="image",
            label_column="labels",
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 0})

    def test_model_init(self):
        scores = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 0})
        model = AutoModelForImageClassification.from_pretrained(self.default_model)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.default_model)
        scores = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="accuracy",
            feature_extractor=feature_extractor,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 0})

    def test_class_init(self):
        evaluator = ImageClassificationEvaluator()
        self.assertEqual(evaluator.task, "image-classification")
        self.assertIsNone(evaluator.default_metric_name)

        scores = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 0})

    def test_default_pipe_init(self):
        scores = self.evaluator.compute(
            data=self.data,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 0})

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
        self.assertEqual(scores, {"accuracy": 0})
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 0})

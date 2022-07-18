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

from datasets import ClassLabel, Dataset, Features, Sequence, Value
from PIL import Image
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from evaluate import (
    ImageClassificationEvaluator,
    TextClassificationEvaluator,
    TokenClassificationEvaluator,
    evaluator,
    load,
)


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


class DummyTokenClassificationPipeline:
    def __init__(self):
        self.task = "token-classification"

    def __call__(self, inputs, **kwargs):
        result = [
            {"start": 0, "entity": "B-LOC"},
            {"start": 2, "entity": "I-LOC"},
            {"start": 4, "entity": "I-LOC"},
            {"start": 9, "entity": "O"},
            {"start": 11, "entity": "O"},
            {"start": 16, "entity": "B-LOC"},
            {"start": 21, "entity": "O"},
        ]

        return [result]


class TestEvaluator(TestCase):
    def test_wrong_task(self):
        self.assertRaises(KeyError, evaluator, "bad_task")


class TestTextClassificationEvaluator(TestCase):
    def setUp(self):
        self.data = Dataset.from_dict({"label": [1, 0], "text": ["great movie", "horrible movie"]})
        self.default_model = "lvwerra/distilbert-imdb"
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
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"f1": 1.0})

    def test_default_pipe_init(self):
        scores = self.evaluator.compute(
            data=self.data,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 1.0})

    def test_overwrite_default_metric(self):
        accuracy = load("accuracy")
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=accuracy,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 1.0})
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
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
                "label": [2, 2],
                "image": [Image.new("RGB", (500, 500), (255, 255, 255)), Image.new("RGB", (500, 500), (170, 95, 170))],
            }
        )
        self.default_model = "lysandre/tiny-vit-random"
        self.pipe = DummyImageClassificationPipeline()
        self.evaluator = evaluator("image-classification")
        self.label_mapping = AutoConfig.from_pretrained(self.default_model).label2id

    def test_pipe_init(self):
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            input_column="image",
            label_column="label",
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 0})

    def test_model_init(self):
        scores = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="accuracy",
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
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 0})

    def test_default_pipe_init(self):
        scores = self.evaluator.compute(
            data=self.data,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 0})

    def test_overwrite_default_metric(self):
        accuracy = load("accuracy")
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=accuracy,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 0})
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
            label_mapping=self.label_mapping,
        )
        self.assertEqual(scores, {"accuracy": 0})


class TestTokenClassificationEvaluator(TestCase):
    def setUp(self):
        features = Features(
            {
                "tokens": Sequence(feature=Value(dtype="string")),
                "ner_tags": Sequence(feature=ClassLabel(names=["O", "B-LOC", "I-LOC"])),
            }
        )

        self.data = Dataset.from_dict(
            {
                "tokens": [["New", "York", "a", "nice", "City", "."]],
                "ner_tags": [[1, 2, 0, 0, 1, 0]],
            },
            features=features,
        )
        self.default_model = "hf-internal-testing/tiny-bert-for-token-classification"
        self.pipe = DummyTokenClassificationPipeline()
        self.evaluator = evaluator("token-classification")

    def test_model_init(self):
        scores = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="seqeval",
        )
        self.assertEqual(scores["overall_accuracy"], 0.5)

        model = AutoModelForTokenClassification.from_pretrained(self.default_model)
        tokenizer = AutoTokenizer.from_pretrained(self.default_model)
        scores = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="seqeval",
            tokenizer=tokenizer,
        )
        self.assertEqual(scores["overall_accuracy"], 0.5)

    def test_class_init(self):
        evaluator = TokenClassificationEvaluator()
        self.assertEqual(evaluator.task, "token-classification")
        self.assertIsNone(evaluator.default_metric_name)

        scores = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="seqeval",
        )
        self.assertEqual(scores["overall_accuracy"], 1.0)

    def test_default_pipe_init(self):
        scores = self.evaluator.compute(
            data=self.data,
        )
        self.assertEqual(scores["overall_accuracy"], 2 / 3)

    def test_overwrite_default_metric(self):
        accuracy = load("seqeval")
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=accuracy,
        )
        self.assertEqual(scores["overall_accuracy"], 1.0)
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="seqeval",
        )
        self.assertEqual(scores["overall_accuracy"], 1.0)

    def test_wrong_task(self):
        self.assertRaises(KeyError, evaluator, "bad_task")

    def test_words_to_offsets(self):
        task_evaluator = evaluator("token-classification")

        words = ["This", "is", "a", "test", "."]
        join_by = " "

        offsets = task_evaluator.words_to_offsets(words, join_by)

        self.assertListEqual([(0, 3), (5, 6), (8, 8), (10, 13), (15, 15)], offsets)

        words = ["日", "本", "語", "はなせるの?"]
        join_by = ""

        offsets = task_evaluator.words_to_offsets(words, join_by)

        self.assertListEqual([(0, 0), (1, 1), (2, 2), (3, 8)], offsets)

    def test_predictions_processor(self):
        task_evaluator = evaluator("token-classification")
        join_by = " "
        words = [["New", "York", "a", "nice", "City", "."]]

        # aligned start and words
        predictions = [
            [
                {"start": 0, "entity": "B-LOC"},
                {"start": 2, "entity": "I-LOC"},
                {"start": 4, "entity": "I-LOC"},
                {"start": 9, "entity": "O"},
                {"start": 11, "entity": "O"},
                {"start": 16, "entity": "B-LOC"},
                {"start": 21, "entity": "O"},
            ]
        ]
        predictions = task_evaluator.predictions_processor(predictions, words, join_by)
        self.assertListEqual(predictions["predictions"][0], ["B-LOC", "I-LOC", "O", "O", "B-LOC", "O"])

        # non-aligned start and words
        predictions = [
            [
                {"start": 0, "entity": "B-LOC"},
                {"start": 2, "entity": "I-LOC"},
                {"start": 9, "entity": "O"},
                {"start": 11, "entity": "O"},
                {"start": 16, "entity": "B-LOC"},
                {"start": 21, "entity": "O"},
            ]
        ]
        predictions = task_evaluator.predictions_processor(predictions, words, join_by)
        self.assertListEqual(predictions["predictions"][0], ["B-LOC", "O", "O", "O", "B-LOC", "O"])

        # non-aligned start and words
        predictions = [
            [
                {"start": 0, "entity": "B-LOC"},
                {"start": 6, "entity": "I-LOC"},
                {"start": 9, "entity": "O"},
                {"start": 11, "entity": "O"},
                {"start": 16, "entity": "B-LOC"},
                {"start": 21, "entity": "O"},
            ]
        ]
        predictions = task_evaluator.predictions_processor(predictions, words, join_by)
        self.assertListEqual(predictions["predictions"][0], ["B-LOC", "O", "O", "O", "B-LOC", "O"])

        # non-aligned start and words
        predictions = [
            [
                {"start": 0, "entity": "B-LOC"},
                {"start": 9, "entity": "O"},
                {"start": 11, "entity": "O"},
                {"start": 16, "entity": "B-LOC"},
                {"start": 21, "entity": "O"},
            ]
        ]
        predictions = task_evaluator.predictions_processor(predictions, words, join_by)
        self.assertListEqual(predictions["predictions"][0], ["B-LOC", "O", "O", "O", "B-LOC", "O"])

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

from time import sleep
from unittest import TestCase

from datasets import Dataset
from PIL import Image
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from evaluate import (
    ImageClassificationEvaluator,
    QuestionAnsweringEvaluator,
    TextClassificationEvaluator,
    evaluator,
    load,
)


class DummyTextClassificationPipeline:
    def __init__(self, sleep_time=None):
        self.task = "text-classification"
        self.sleep_time = sleep_time

    def __call__(self, text, **kwargs):
        if self.sleep_time is not None:
            sleep(self.sleep_time)
        return [{"label": "NEGATIVE"} if i % 2 == 1 else {"label": "POSITIVE"} for i, _ in enumerate(text)]


class DummyImageClassificationPipeline:
    def __init__(self):
        self.task = "image-classification"

    def __call__(self, images, **kwargs):
        return [[{"score": 0.9, "label": "yurt"}, {"score": 0.1, "label": "umbrella"}] for i, _ in enumerate(images)]


class DummyQuestionAnsweringPipeline:
    def __init__(self, v2: bool):
        self.task = "question-answering"
        self.v2 = v2

    def __call__(self, question, context, **kwargs):
        if self.v2:
            return [
                {"score": 0.95, "start": 31, "end": 39, "answer": "Felix"}
                if i % 2 == 0
                else {"score": 0.95, "start": 0, "end": 0, "answer": ""}
                for i in range(len(question))
            ]
        else:
            return [{"score": 0.95, "start": 31, "end": 39, "answer": "Felix"} for _ in question]


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
        self.perf_pipe = DummyTextClassificationPipeline(sleep_time=0.1)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.default_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.default_model)
        self.evaluator = evaluator("text-classification")
        self.label_mapping = {"NEGATIVE": 0.0, "POSITIVE": 1.0}

    def test_pipe_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            input_column="text",
            label_column="label",
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 1.0)

    def test_model_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 1.0)
        results = self.evaluator.compute(
            model_or_pipeline=self.model,
            data=self.data,
            metric="accuracy",
            tokenizer=self.tokenizer,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 1.0)

    def test_class_init(self):
        evaluator = TextClassificationEvaluator()
        self.assertEqual(evaluator.task, "text-classification")
        self.assertIsNone(evaluator.default_metric_name)

        results = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="f1",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["f1"], 1.0)

    def test_default_pipe_init(self):
        results = self.evaluator.compute(
            data=self.data,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 1.0)

    def test_overwrite_default_metric(self):
        accuracy = load("accuracy")
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=accuracy,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 1.0)
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 1.0)

    def test_bootstrap(self):
        data = Dataset.from_dict({"label": [1, 0, 0], "text": ["great movie", "great movie", "horrible movie"]})

        results = self.evaluator.compute(
            model_or_pipeline=self.model,
            data=data,
            metric="accuracy",
            tokenizer=self.tokenizer,
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

    def test_perf(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.perf_pipe,
            data=self.data,
            metric="accuracy",
            tokenizer=self.tokenizer,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
            n_resamples=10,
            random_state=0,
        )
        self.assertEqual(results["accuracy"], 1.0)
        self.assertAlmostEqual(results["total_time_in_seconds"], 0.1, 1)
        self.assertAlmostEqual(results["samples_per_second"], len(self.data) / results["total_time_in_seconds"], 5)
        self.assertAlmostEqual(results["latency_in_seconds"], results["total_time_in_seconds"] / len(self.data), 5)

    def test_bootstrap_and_perf(self):
        data = Dataset.from_dict({"label": [1, 0, 0], "text": ["great movie", "great movie", "horrible movie"]})

        results = self.evaluator.compute(
            model_or_pipeline=self.perf_pipe,
            data=data,
            metric="accuracy",
            tokenizer=self.tokenizer,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
            strategy="bootstrap",
            n_resamples=10,
            random_state=0,
        )
        self.assertAlmostEqual(results["accuracy"]["score"], 0.666666, 5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][0], 0.333333, 5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][1], 0.666666, 5)
        self.assertAlmostEqual(results["accuracy"]["standard_error"], 0.22498285, 5)
        self.assertAlmostEqual(results["total_time_in_seconds"], 0.1, 1)
        self.assertAlmostEqual(results["samples_per_second"], len(data) / results["total_time_in_seconds"], 5)
        self.assertAlmostEqual(results["latency_in_seconds"], results["total_time_in_seconds"] / len(data), 5)


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
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            input_column="image",
            label_column="labels",
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

    def test_model_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)
        model = AutoModelForImageClassification.from_pretrained(self.default_model)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.default_model)
        results = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="accuracy",
            feature_extractor=feature_extractor,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

    def test_class_init(self):
        evaluator = ImageClassificationEvaluator()
        self.assertEqual(evaluator.task, "image-classification")
        self.assertIsNone(evaluator.default_metric_name)

        results = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

    def test_default_pipe_init(self):
        results = self.evaluator.compute(
            data=self.data,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

    def test_overwrite_default_metric(self):
        accuracy = load("accuracy")
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=accuracy,
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)


class TestQuestionAnsweringEvaluator(TestCase):
    def setUp(self):
        self.data = Dataset.from_dict(
            {
                "id": ["56be4db0acb8001400a502ec", "56be4db0acb8001400a502ed"],
                "context": ["My name is Felix and I love cookies!", "Misa name is Felix and misa love cookies!"],
                "answers": [{"text": ["Felix"], "answer_start": [11]}, {"text": ["Felix"], "answer_start": [13]}],
                "question": ["What is my name?", "What is my name?"],
            }
        )
        self.data_v2 = Dataset.from_dict(
            {
                "id": ["56be4db0acb8001400a502ec", "56be4db0acb8001400a502ed"],
                "context": ["My name is Felix and I love cookies!", "Let's explore the city!"],
                "answers": [{"text": ["Felix"], "answer_start": [11]}, {"text": [], "answer_start": []}],
                "question": ["What is my name?", "What is my name?"],
            }
        )

        self.default_model = "mrm8488/bert-tiny-finetuned-squadv2"
        self.pipe = DummyQuestionAnsweringPipeline(v2=False)
        self.pipe_v2 = DummyQuestionAnsweringPipeline(v2=True)
        self.evaluator = evaluator("question-answering")

    def test_pipe_init(self):
        # squad_v1-like dataset
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
        )
        self.assertEqual(scores["exact_match"], 100.0)
        self.assertEqual(scores["f1"], 100.0)

    def test_model_init(self):
        # squad_v1-like dataset
        scores = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="squad",
        )
        self.assertEqual(scores["exact_match"], 0)
        self.assertEqual(scores["f1"], 0)

        model = AutoModelForQuestionAnswering.from_pretrained(self.default_model)
        tokenizer = AutoTokenizer.from_pretrained(self.default_model)
        scores = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="squad",
            tokenizer=tokenizer,
        )
        self.assertEqual(scores["exact_match"], 0)
        self.assertEqual(scores["f1"], 0)

    def test_class_init(self):
        # squad_v1-like dataset
        evaluator = QuestionAnsweringEvaluator()
        self.assertEqual(evaluator.task, "question-answering")
        self.assertIsNone(evaluator.default_metric_name)

        scores = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="squad",
        )
        self.assertEqual(scores["exact_match"], 100.0)
        self.assertEqual(scores["f1"], 100.0)

        # squad_v2-like dataset
        evaluator = QuestionAnsweringEvaluator()
        self.assertEqual(evaluator.task, "question-answering")
        self.assertIsNone(evaluator.default_metric_name)

        scores = evaluator.compute(
            model_or_pipeline=self.pipe_v2,
            data=self.data_v2,
            metric="squad_v2",
        )
        self.assertDictEqual(
            {key: scores[key] for key in ["HasAns_f1", "NoAns_f1"]}, {"HasAns_f1": 100.0, "NoAns_f1": 100.0}
        )

    def test_default_pipe_init(self):
        # squad_v1-like dataset
        scores = self.evaluator.compute(
            data=self.data,
        )
        self.assertEqual(scores["exact_match"], 100.0)
        self.assertEqual(scores["f1"], 100.0)

        # squad_v2-like dataset
        scores = self.evaluator.compute(
            data=self.data_v2,
            metric="squad_v2",
        )
        self.assertDictEqual(
            {key: scores[key] for key in ["HasAns_f1", "NoAns_f1"]}, {"HasAns_f1": 100.0, "NoAns_f1": 0.0}
        )

    def test_overwrite_default_metric(self):
        # squad_v1-like dataset
        squad = load("squad")
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=squad,
        )
        self.assertEqual(scores["exact_match"], 100.0)
        self.assertEqual(scores["f1"], 100.0)

        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="squad",
        )
        self.assertEqual(scores["exact_match"], 100.0)
        self.assertEqual(scores["f1"], 100.0)

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

import importlib
import os
import tempfile
import warnings
from numbers import Number
from time import sleep
from unittest import TestCase, mock

import numpy as np
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from PIL import Image
from scipy.stats import bootstrap
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForImageClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from evaluate import (
    AudioClassificationEvaluator,
    AutomaticSpeechRecognitionEvaluator,
    Evaluator,
    ImageClassificationEvaluator,
    QuestionAnsweringEvaluator,
    Text2TextGenerationEvaluator,
    TextClassificationEvaluator,
    TextGenerationEvaluator,
    TokenClassificationEvaluator,
    evaluator,
    load,
)

from .utils import slow


class DummyTextGenerationPipeline:
    def __init__(self, prefix="generated", task="text-generation", num_return_sequences=1):
        self.task = task
        self.prefix = prefix
        self.num_return_sequences = num_return_sequences

    def __call__(self, inputs, **kwargs):
        return [[{f"{self.prefix}_text": "Lorem ipsum"} for _ in range(self.num_return_sequences)] for _ in inputs]


class DummyText2TextGenerationPipeline:
    def __init__(self, prefix="generated", task="text2text-generation"):
        self.task = task
        self.prefix = prefix

    def __call__(self, inputs, **kwargs):
        return [{f"{self.prefix}_text": "Lorem ipsum"} for _ in inputs]


class DummyTextClassificationPipeline:
    def __init__(self, sleep_time=None):
        self.task = "text-classification"
        self.sleep_time = sleep_time

    def __call__(self, inputs, **kwargs):
        if self.sleep_time is not None:
            sleep(self.sleep_time)
        return [{"label": "NEGATIVE"} if i % 2 == 1 else {"label": "POSITIVE"} for i, _ in enumerate(inputs)]


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


class BatchedDummyTokenClassificationPipeline(DummyTokenClassificationPipeline):
    """Repeat the existing dummy output once per input example."""

    def __call__(self, inputs, **kwargs):
        parent_call = super().__call__
        return [parent_call([text], **kwargs)[0] for text in inputs]


class DummyAutomaticSpeechRecognitionPipeline:
    def __init__(self) -> None:
        self.task = "automatic-speech-recognition"

    def __call__(self, inputs, **kwargs):
        return [{"text": "Lorem ipsum"} for _ in inputs]


class DummyAudioClassificationPipeline:
    def __init__(self):
        self.task = "audio-classification"

    def __call__(self, audio, **kwargs):
        return [[{"score": 0.9, "label": "yes"}, {"score": 0.1, "label": "no"}] for i, _ in enumerate(audio)]


class TestEvaluator(TestCase):
    def setUp(self):
        self.data = Dataset.from_dict({"label": [1, 0], "text": ["great movie", "horrible movie"]})
        self.default_ckpt = "hf-internal-testing/tiny-random-bert"
        self.default_model = AutoModelForSequenceClassification.from_pretrained(self.default_ckpt, num_labels=2)
        self.default_tokenizer = AutoTokenizer.from_pretrained(self.default_ckpt)
        self.pipe = pipeline("text-classification", model=self.default_model, tokenizer=self.default_tokenizer)
        self.evaluator = evaluator("text-classification")
        self.data = Dataset.from_dict({"label": [1, 0], "text": ["great movie", "horrible movie"]})
        self.label_mapping = {"LABEL_0": 0.0, "LABEL_1": 1.0}

    def test_wrong_task(self):
        self.assertRaises(KeyError, evaluator, "bad_task")

    def test_device_placement(self):
        orig_import = __import__

        pt_mock = mock.Mock()
        tf_mock = mock.Mock()

        # mock import of torch and tensorflow
        def import_pt_tf_mock(name, *args):
            if name == "torch":
                if pt_available:
                    return pt_mock
                else:
                    raise ImportError
            if name == "tensorflow":
                if tf_available:
                    return tf_mock
                else:
                    raise ImportError
            return orig_import(name, *args)

        with mock.patch("builtins.__import__", side_effect=import_pt_tf_mock):
            # neither pt or tf are available
            pt_available = False
            tf_available = False
            self.assertEqual(Evaluator._infer_device(), -1)

            # pt available but no GPU
            pt_available = True
            pt_mock.cuda.is_available.return_value = False
            self.assertEqual(Evaluator._infer_device(), -1)

            # pt available and GPU found
            pt_mock.cuda.is_available.return_value = True
            self.assertEqual(Evaluator._infer_device(), 0)

            # tf available but no GPU
            pt_available = False
            tf_available = True
            tf_mock.config.list_physical_devices.return_value = []
            self.assertEqual(Evaluator._infer_device(), -1)

            # tf available and GPU found
            tf_mock.config.list_physical_devices.return_value = ["GPU:0", "GPU:1"]
            self.assertEqual(Evaluator._infer_device(), 0)

            # pt accelerator found and pipeline instantiated on CPU
            pt_mock.cuda.is_available.return_value = True
            self.assertRaises(
                ValueError, Evaluator.check_for_mismatch_in_device_setup, Evaluator._infer_device(), self.pipe
            )

            # tf accelerator found and pipeline instantiated on CPU
            pt_available = False
            tf_available = True
            self.assertRaises(
                ValueError, Evaluator.check_for_mismatch_in_device_setup, Evaluator._infer_device(), self.pipe
            )

    def test_pipe_init(self):
        self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            input_column="text",
            label_column="label",
            label_mapping=self.label_mapping,
        )

    def test_model_init(self):
        self.evaluator.compute(
            model_or_pipeline=self.default_model,
            tokenizer=self.default_tokenizer,
            data=self.data,
            input_column="text",
            label_column="label",
            label_mapping=self.label_mapping,
        )

    def test_model_str_init(self):
        self.evaluator.compute(
            model_or_pipeline=self.default_ckpt,
            data=self.data,
            input_column="text",
            label_column="label",
            label_mapping=self.label_mapping,
        )


class TestTextClassificationEvaluator(TestCase):
    def setUp(self):
        self.data = Dataset.from_dict({"label": [1, 0], "text": ["great movie", "horrible movie"]})
        self.default_model = "lvwerra/distilbert-imdb"
        self.input_column = "text"
        self.label_column = "label"
        self.pipe = DummyTextClassificationPipeline()
        self.perf_pipe = DummyTextClassificationPipeline(sleep_time=0.1)
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

    @slow
    def test_model_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
        )

        model = AutoModelForSequenceClassification.from_pretrained(self.default_model)
        tokenizer = AutoTokenizer.from_pretrained(self.default_model)

        self.assertEqual(results["accuracy"], 1.0)
        results = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="accuracy",
            tokenizer=tokenizer,
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
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["f1"], 1.0)

    @slow
    def test_default_pipe_init(self):
        results = self.evaluator.compute(
            data=self.data,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 1.0)

    def test_data_loading(self):

        # Test passing in dataset by name with split
        data = self.evaluator.load_data("evaluate/imdb-ci", split="test[:1]")
        self.evaluator.prepare_data(data=data, input_column="text", label_column="label", second_input_column=None)

        # Test passing in dataset by name without split and inferring the optimal split
        data = self.evaluator.load_data("evaluate/imdb-ci")
        self.evaluator.prepare_data(data=data, input_column="text", label_column="label", second_input_column=None)

        # Test that it chooses the correct one (e.g. imdb only has train and test, but no validation)
        self.assertEqual(data.split, "test")

        # Test that the data point returned is correct; this maps to the first example in the dataset
        self.assertEqual(data[0]["text"], "I love movies about whales!")

        # Test loading subset of a dataset with the `name` field
        data = self.evaluator.load_data("evaluate/glue-ci", subset="cola", split="test")
        self.assertEqual(isinstance(data, Dataset), True)

        # Test loading subset of a dataset with the `name` field and having it infer the split
        data = self.evaluator.load_data("evaluate/glue-ci", subset="cola")
        self.assertEqual(isinstance(data, Dataset), True)

    def test_overwrite_default_metric(self):
        accuracy = load("accuracy")
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=accuracy,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 1.0)
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 1.0)

    def test_bootstrap(self):
        data = Dataset.from_dict({"label": [1, 0, 0], "text": ["great movie", "great movie", "horrible movie"]})

        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=data,
            metric="accuracy",
            label_mapping=self.label_mapping,
            strategy="bootstrap",
            n_resamples=10,
            random_state=0,
        )
        self.assertAlmostEqual(results["accuracy"]["score"], 0.666666, 5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][0], 0.33557, 5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][1], 1.0, 5)
        self.assertAlmostEqual(results["accuracy"]["standard_error"], 0.22498, 5)

    def test_perf(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.perf_pipe,
            data=self.data,
            metric="accuracy",
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
            input_column=self.input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping,
            strategy="bootstrap",
            n_resamples=10,
            random_state=0,
        )
        self.assertAlmostEqual(results["accuracy"]["score"], 0.666666, 5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][0], 0.33557, 5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][1], 1.0, 5)
        self.assertAlmostEqual(results["accuracy"]["standard_error"], 0.22498285, 5)
        self.assertAlmostEqual(results["total_time_in_seconds"], 0.1, 1)
        self.assertAlmostEqual(results["samples_per_second"], len(data) / results["total_time_in_seconds"], 5)
        self.assertAlmostEqual(results["latency_in_seconds"], results["total_time_in_seconds"] / len(data), 5)


class TestTextClassificationEvaluatorTwoColumns(TestCase):
    def setUp(self):
        self.data = Dataset.from_dict(
            {
                "label": [1, 0],
                "premise": ["great car", "great movie"],
                "hypothesis": ["great vehicle", "horrible movie"],
            }
        )
        self.default_model = "prajjwal1/bert-tiny-mnli"
        self.input_column = "premise"
        self.second_input_column = "hypothesis"
        self.label_column = "label"
        self.pipe = DummyTextClassificationPipeline()
        self.evaluator = evaluator("text-classification")
        self.label_mapping = {"NEGATIVE": 0.0, "POSITIVE": 1.0}
        self.label_mapping2 = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}

    def test_pipe_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            input_column=self.input_column,
            second_input_column=self.second_input_column,
            label_column="label",
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 1.0)

    @slow
    def test_model_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            second_input_column=self.second_input_column,
            label_column=self.label_column,
            label_mapping=self.label_mapping2,
        )
        self.assertEqual(results["accuracy"], 1.0)

        model = AutoModelForSequenceClassification.from_pretrained(self.default_model)
        tokenizer = AutoTokenizer.from_pretrained(self.default_model)

        results = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="accuracy",
            input_column=self.input_column,
            second_input_column=self.second_input_column,
            tokenizer=tokenizer,
            label_mapping=self.label_mapping2,
        )
        self.assertEqual(results["accuracy"], 1.0)


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
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

    @slow
    def test_model_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="accuracy",
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
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

    @slow
    def test_default_pipe_init(self):
        results = self.evaluator.compute(
            data=self.data,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

    def test_overwrite_default_metric(self):
        accuracy = load("accuracy")
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=accuracy,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
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
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
        )
        self.assertEqual(results["exact_match"], 100.0)
        self.assertEqual(results["f1"], 100.0)

    @slow
    def test_model_init(self):
        # squad_v1-like dataset
        results = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="squad",
        )
        self.assertEqual(results["exact_match"], 0)
        self.assertEqual(results["f1"], 100 / 3)

        model = AutoModelForQuestionAnswering.from_pretrained(self.default_model)
        tokenizer = AutoTokenizer.from_pretrained(self.default_model)

        results = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="squad",
            tokenizer=tokenizer,
        )
        self.assertEqual(results["exact_match"], 0)
        self.assertEqual(results["f1"], 100 / 3)

    def test_class_init(self):
        # squad_v1-like dataset
        evaluator = QuestionAnsweringEvaluator()
        self.assertEqual(evaluator.task, "question-answering")
        self.assertIsNone(evaluator.default_metric_name)

        results = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="squad",
        )
        self.assertEqual(results["exact_match"], 100.0)
        self.assertEqual(results["f1"], 100.0)

        # squad_v2-like dataset
        evaluator = QuestionAnsweringEvaluator()
        self.assertEqual(evaluator.task, "question-answering")
        self.assertIsNone(evaluator.default_metric_name)

        results = evaluator.compute(
            model_or_pipeline=self.pipe_v2,
            data=self.data_v2,
            metric="squad_v2",
        )
        self.assertDictEqual(
            {key: results[key] for key in ["HasAns_f1", "NoAns_f1"]}, {"HasAns_f1": 100.0, "NoAns_f1": 100.0}
        )

    @slow
    def test_default_pipe_init(self):
        # squad_v1-like dataset
        results = self.evaluator.compute(
            data=self.data,
        )
        self.assertEqual(results["exact_match"], 100.0)
        self.assertEqual(results["f1"], 100.0)

        # squad_v2-like dataset
        results = self.evaluator.compute(
            data=self.data_v2,
            metric="squad_v2",
        )
        self.assertDictEqual(
            {key: results[key] for key in ["HasAns_f1", "NoAns_f1"]}, {"HasAns_f1": 100.0, "NoAns_f1": 0.0}
        )

    def test_data_loading(self):
        # Test passing in dataset by name with data_split
        data = self.evaluator.load_data("evaluate/squad-ci", split="validation[:1]")
        self.evaluator.prepare_data(
            data=data, question_column="question", context_column="context", id_column="id", label_column="answers"
        )

        # Test passing in dataset by name without data_split and inferring the optimal split
        data = self.evaluator.load_data("evaluate/squad-ci")
        self.evaluator.prepare_data(
            data=data, question_column="question", context_column="context", id_column="id", label_column="answers"
        )

        # Test that it chooses the correct one (e.g. squad only has train and validation, but no test)
        self.assertEqual(data.split, "validation")

        # Test that the data point returned is correct; this maps to the first example in the squad-ci dataset
        self.assertEqual(data[0]["id"], "56be4db0acb8001400a502ec")

    def test_overwrite_default_metric(self):
        # squad_v1-like dataset
        squad = load("squad")
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=squad,
        )
        self.assertEqual(results["exact_match"], 100.0)
        self.assertEqual(results["f1"], 100.0)

        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="squad",
        )
        self.assertEqual(results["exact_match"], 100.0)
        self.assertEqual(results["f1"], 100.0)


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

    @slow
    def test_model_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="seqeval",
        )
        self.assertEqual(results["overall_accuracy"], 0.5)

        model = AutoModelForTokenClassification.from_pretrained(self.default_model)
        tokenizer = AutoTokenizer.from_pretrained(self.default_model)
        results = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="seqeval",
            tokenizer=tokenizer,
        )
        self.assertEqual(results["overall_accuracy"], 0.5)

    def test_class_init(self):
        evaluator = TokenClassificationEvaluator()
        self.assertEqual(evaluator.task, "token-classification")
        self.assertIsNone(evaluator.default_metric_name)

        results = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="seqeval",
        )
        self.assertEqual(results["overall_accuracy"], 1.0)

    @slow
    def test_default_pipe_init(self):
        results = self.evaluator.compute(
            data=self.data,
        )
        self.assertEqual(results["overall_accuracy"], 2 / 3)

    def test_overwrite_default_metric(self):
        accuracy = load("seqeval")
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=accuracy,
        )
        self.assertEqual(results["overall_accuracy"], 1.0)
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="seqeval",
        )
        self.assertEqual(results["overall_accuracy"], 1.0)

    def test_data_loading(self):
        # Test passing in dataset by name with data_split
        data = self.evaluator.load_data("evaluate/conll2003-ci", split="validation[:1]")
        self.evaluator.prepare_data(
            data=data,
            input_column="tokens",
            label_column="ner_tags",
            join_by=" ",
        )

        # Test passing in dataset by name without data_split and inferring the optimal split
        data = self.evaluator.load_data("evaluate/conll2003-ci")
        self.evaluator.prepare_data(
            data=data,
            input_column="tokens",
            label_column="ner_tags",
            join_by=" ",
        )

        # Test that it chooses the correct one (e.g. conll2003 has train, validation, test but should select test)
        self.assertEqual(data.split, "test")

        # Test that the data point returned is correct; this maps to the first example in the dataset
        self.assertEqual(data[0]["id"], "0")

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


class TestTokenClassificationBootstrap(TestCase):
    """Regression coverage for sentence-level token-classification bootstrap (#546)."""

    RAGGED_METRIC_INPUTS = {
        "references": [
            ["O"],
            ["B-X", "O"],
            ["O", "B-X", "I-X"],
            ["B-X"],
            ["O", "O"],
            ["B-X", "I-X", "O"],
            ["O"],
            ["O", "B-X"],
        ],
        "predictions": [
            ["O"],
            ["O", "O"],
            ["O", "B-X", "I-X"],
            ["O"],
            ["O", "O"],
            ["B-X", "O", "O"],
            ["O"],
            ["O", "O"],
        ],
    }
    EQUAL_METRIC_INPUTS = {
        "references": [
            ["O", "O", "O"],
            ["B-X", "O", "O"],
            ["O", "B-X", "I-X"],
            ["B-X", "O", "O"],
            ["O", "O", "O"],
            ["B-X", "I-X", "O"],
            ["O", "O", "O"],
            ["O", "B-X", "O"],
        ],
        "predictions": [
            ["O", "O", "O"],
            ["O", "O", "O"],
            ["O", "B-X", "I-X"],
            ["O", "O", "O"],
            ["O", "O", "O"],
            ["B-X", "O", "O"],
            ["O", "O", "O"],
            ["O", "O", "O"],
        ],
    }
    VARIABLE_LENGTH_TOKENS = [
        ["New", "York", "a", "nice", "City", "."],
        ["New", "York"],
        ["New", "York", "a"],
        ["New", "York", "a", "nice"],
        ["New", "York", "a", "nice", "City", "."],
        ["New", "York"],
        ["New", "York", "a"],
        ["New", "York", "a", "nice", "City"],
    ]
    VARIABLE_LENGTH_REFERENCES = [
        ["B-LOC", "I-LOC", "O", "O", "B-LOC", "O"],
        ["O", "O"],
        ["B-LOC", "I-LOC", "O"],
        ["O", "O", "O", "O"],
        ["O", "O", "O", "O", "B-LOC", "O"],
        ["B-LOC", "I-LOC"],
        ["O", "O", "O"],
        ["B-LOC", "I-LOC", "O", "O", "O"],
    ]
    EQUAL_LENGTH_TOKENS = [["New", "York", "a"]] * 8
    EQUAL_LENGTH_REFERENCES = [
        ["B-LOC", "I-LOC", "O"],
        ["O", "O", "O"],
        ["B-LOC", "I-LOC", "O"],
        ["O", "O", "O"],
        ["O", "B-LOC", "O"],
        ["B-LOC", "I-LOC", "O"],
        ["O", "O", "O"],
        ["O", "O", "B-LOC"],
    ]
    PERFORMANCE_KEYS = ("total_time_in_seconds", "samples_per_second", "latency_in_seconds")
    ENTITY_RATE_KEYS = ("precision", "recall", "f1")

    def setUp(self):
        self._metric_cache = tempfile.TemporaryDirectory()
        self.addCleanup(self._metric_cache.cleanup)
        self._seqeval_counter = 0
        self.evaluator = TokenClassificationEvaluator()
        self.pipe = DummyTokenClassificationPipeline()
        self.batched_pipe = BatchedDummyTokenClassificationPipeline()
        features = Features(
            {
                "tokens": Sequence(feature=Value(dtype="string")),
                "ner_tags": Sequence(feature=ClassLabel(names=["O", "B-LOC", "I-LOC"])),
            }
        )
        self.single_example_data = Dataset.from_dict(
            {
                "tokens": [["New", "York", "a", "nice", "City", "."]],
                "ner_tags": [[1, 2, 0, 0, 1, 0]],
            },
            features=features,
        )

    def _load_local_seqeval(self):
        self._seqeval_counter += 1
        metric_path = os.path.join(os.path.dirname(__file__), "..", "metrics", "seqeval")
        return load(
            metric_path,
            cache_dir=self._metric_cache.name,
            keep_in_memory=True,
            experiment_id="token-bootstrap-{0}".format(self._seqeval_counter),
        )

    def _string_tag_dataset(self, tokens, references):
        features = Features(
            {
                "tokens": Sequence(feature=Value(dtype="string")),
                "ner_tags": Sequence(feature=Value(dtype="string")),
            }
        )
        return Dataset.from_dict({"tokens": tokens, "ner_tags": references}, features=features)

    def _independent_sentence_bootstrap(
        self, metric, metric_inputs, path, confidence_level, n_resamples, random_state, metric_kwargs=None
    ):
        metric_kwargs = {} if metric_kwargs is None else metric_kwargs
        n_observations = len(next(iter(metric_inputs.values())))

        def statistic(indices):
            sampled = {column: [rows[int(index)] for index in indices] for column, rows in metric_inputs.items()}
            output = metric.compute(**sampled, **metric_kwargs)
            current = output
            for key in path:
                if key not in current:
                    return float("nan")
                current = current[key]
            return current

        return bootstrap(
            (np.arange(n_observations),),
            statistic,
            paired=True,
            vectorized=False,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )

    def _assert_scalar_bootstrap_record(self, record, score):
        self.assertIsInstance(record, dict)
        self.assertEqual(set(record), {"score", "confidence_interval", "standard_error"})
        self.assertEqual(record["score"], score)
        self.assertEqual(len(record["confidence_interval"]), 2)
        self.assertIsInstance(record["standard_error"], Number)

    def _assert_entity_bootstrap_record(self, record, ordinary_entity):
        self.assertIsInstance(record, dict)
        self.assertEqual(set(record), {"score", "confidence_interval", "standard_error"})
        self.assertEqual(record["score"], ordinary_entity)
        self.assertEqual(set(record["confidence_interval"]), set(self.ENTITY_RATE_KEYS))
        self.assertEqual(set(record["standard_error"]), set(self.ENTITY_RATE_KEYS))
        self.assertIn("number", record["score"])
        self.assertNotIn("number", record["confidence_interval"])
        self.assertNotIn("number", record["standard_error"])
        for rate_key in self.ENTITY_RATE_KEYS:
            self.assertEqual(len(record["confidence_interval"][rate_key]), 2)
            self.assertIsInstance(record["standard_error"][rate_key], Number)

    def _assert_bootstrap_result_matches_ordinary(self, bootstrap_result, ordinary):
        self.assertEqual(set(bootstrap_result), set(ordinary))
        for key, value in ordinary.items():
            if isinstance(value, dict):
                self._assert_entity_bootstrap_record(bootstrap_result[key], value)
            else:
                self._assert_scalar_bootstrap_record(bootstrap_result[key], value)

    def _assert_finite_record_interval(self, record):
        interval = record["confidence_interval"]
        if isinstance(interval, dict):
            for rate_key in self.ENTITY_RATE_KEYS:
                self.assertTrue(np.all(np.isfinite(interval[rate_key])))
                self.assertTrue(np.isfinite(record["standard_error"][rate_key]))
        else:
            self.assertTrue(np.all(np.isfinite(interval)))
            self.assertTrue(np.isfinite(record["standard_error"]))

    def _assert_matches_independent_bootstrap(
        self, record, metric, metric_inputs, path, confidence_level, n_resamples, random_state, metric_kwargs=None
    ):
        reference = self._independent_sentence_bootstrap(
            metric,
            metric_inputs,
            path,
            confidence_level,
            n_resamples,
            random_state,
            metric_kwargs=metric_kwargs,
        )
        np.testing.assert_allclose(
            record["confidence_interval"],
            (reference.confidence_interval.low, reference.confidence_interval.high),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            record["standard_error"], reference.standard_error, rtol=1e-12, atol=1e-12, equal_nan=True
        )

    def _assert_performance_keys(self, result):
        for key in self.PERFORMANCE_KEYS:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], Number)
            self.assertNotIsInstance(result[key], dict)

    def test_simple_scoring_unchanged_on_existing_fixture(self):
        metric = self._load_local_seqeval()
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe, data=self.single_example_data, metric=metric, device=-1
        )
        self.assertEqual(results["overall_accuracy"], 1.0)
        self.assertEqual(results["overall_f1"], 1.0)
        self.assertEqual(results["LOC"]["number"], 2)
        self.assertIsInstance(results["LOC"], dict)
        self.assertIsInstance(results["overall_accuracy"], Number)
        self.assertNotIsInstance(results["overall_accuracy"], dict)
        self._assert_performance_keys(results)

    def test_bootstrap_rejects_fewer_than_two_examples(self):
        metric = self._load_local_seqeval()
        with self.assertRaises(ValueError) as context:
            self.evaluator.compute(
                model_or_pipeline=self.pipe,
                data=self.single_example_data,
                metric=metric,
                device=-1,
                strategy="bootstrap",
                n_resamples=9,
                random_state=42,
            )
        self.assertIn("two or more observations", str(context.exception))

    def test_bootstrap_variable_length_full_evaluator(self):
        metric = self._load_local_seqeval()
        data = self._string_tag_dataset(self.VARIABLE_LENGTH_TOKENS, self.VARIABLE_LENGTH_REFERENCES)
        simple = self.evaluator.compute(model_or_pipeline=self.batched_pipe, data=data, metric=metric, device=-1)
        self.assertAlmostEqual(simple["overall_f1"], 0.7058823529411764)
        bootstrap_result = self.evaluator.compute(
            model_or_pipeline=self.batched_pipe,
            data=data,
            metric=metric,
            device=-1,
            strategy="bootstrap",
            n_resamples=99,
            random_state=42,
        )
        ordinary_keys = {key: simple[key] for key in simple if key not in self.PERFORMANCE_KEYS}
        self._assert_bootstrap_result_matches_ordinary(
            {key: bootstrap_result[key] for key in ordinary_keys}, ordinary_keys
        )
        self._assert_performance_keys(bootstrap_result)
        self._assert_finite_record_interval(bootstrap_result["overall_f1"])
        self._assert_finite_record_interval(bootstrap_result["overall_accuracy"])
        # Dummy alignments can make the LOC rate statistic degenerate; overall rates stay finite.

        metric_inputs, pipe_inputs = self.evaluator.prepare_data(data, "tokens", "ner_tags", " ")
        predictions, _ = self.evaluator.call_pipeline(self.batched_pipe, pipe_inputs)
        metric_inputs.update(self.evaluator.predictions_processor(predictions, data["tokens"], " "))
        oracle_metric = self._load_local_seqeval()
        self._assert_matches_independent_bootstrap(
            bootstrap_result["overall_f1"], oracle_metric, metric_inputs, ("overall_f1",), 0.95, 99, 42
        )
        self._assert_matches_independent_bootstrap(
            bootstrap_result["overall_accuracy"], oracle_metric, metric_inputs, ("overall_accuracy",), 0.95, 99, 42
        )
        self._assert_matches_independent_bootstrap(
            {
                "confidence_interval": bootstrap_result["LOC"]["confidence_interval"]["f1"],
                "standard_error": bootstrap_result["LOC"]["standard_error"]["f1"],
            },
            oracle_metric,
            metric_inputs,
            ("LOC", "f1"),
            0.95,
            99,
            42,
        )

    def test_bootstrap_equal_length_full_evaluator(self):
        metric = self._load_local_seqeval()
        data = self._string_tag_dataset(self.EQUAL_LENGTH_TOKENS, self.EQUAL_LENGTH_REFERENCES)
        simple = self.evaluator.compute(model_or_pipeline=self.batched_pipe, data=data, metric=metric, device=-1)
        bootstrap_result = self.evaluator.compute(
            model_or_pipeline=self.batched_pipe,
            data=data,
            metric=metric,
            device=-1,
            strategy="bootstrap",
            n_resamples=99,
            random_state=42,
        )
        ordinary_keys = {key: simple[key] for key in simple if key not in self.PERFORMANCE_KEYS}
        self._assert_bootstrap_result_matches_ordinary(
            {key: bootstrap_result[key] for key in ordinary_keys}, ordinary_keys
        )
        self._assert_performance_keys(bootstrap_result)
        self._assert_finite_record_interval(bootstrap_result["overall_f1"])
        self._assert_finite_record_interval(bootstrap_result["LOC"])
        self.assertGreater(simple["LOC"]["number"], 0)

        metric_inputs, pipe_inputs = self.evaluator.prepare_data(data, "tokens", "ner_tags", " ")
        predictions, _ = self.evaluator.call_pipeline(self.batched_pipe, pipe_inputs)
        metric_inputs.update(self.evaluator.predictions_processor(predictions, data["tokens"], " "))
        oracle_metric = self._load_local_seqeval()
        self._assert_matches_independent_bootstrap(
            bootstrap_result["overall_f1"], oracle_metric, metric_inputs, ("overall_f1",), 0.95, 99, 42
        )

    def test_bootstrap_metric_stage_matches_sentence_index_oracle(self):
        for fixture in (self.RAGGED_METRIC_INPUTS, self.EQUAL_METRIC_INPUTS):
            metric = self._load_local_seqeval()
            ordinary = metric.compute(**fixture)
            result = self.evaluator.compute_metric(
                metric, fixture, strategy="bootstrap", n_resamples=99, random_state=42
            )
            self._assert_bootstrap_result_matches_ordinary(result, ordinary)
            self._assert_finite_record_interval(result["overall_f1"])
            self._assert_finite_record_interval(result["overall_accuracy"])
            self._assert_finite_record_interval(result["X"])
            oracle_metric = self._load_local_seqeval()
            self._assert_matches_independent_bootstrap(
                result["overall_f1"], oracle_metric, fixture, ("overall_f1",), 0.95, 99, 42
            )
            self._assert_matches_independent_bootstrap(
                result["overall_accuracy"], oracle_metric, fixture, ("overall_accuracy",), 0.95, 99, 42
            )
            self._assert_matches_independent_bootstrap(
                {
                    "confidence_interval": result["X"]["confidence_interval"]["f1"],
                    "standard_error": result["X"]["standard_error"]["f1"],
                },
                oracle_metric,
                fixture,
                ("X", "f1"),
                0.95,
                99,
                42,
            )
            self._assert_matches_independent_bootstrap(
                {
                    "confidence_interval": result["X"]["confidence_interval"]["precision"],
                    "standard_error": result["X"]["standard_error"]["precision"],
                },
                oracle_metric,
                fixture,
                ("X", "precision"),
                0.95,
                99,
                42,
            )

    def test_recording_metric_preserves_rows_and_jackknife(self):
        original = {
            "predictions": [["O", "B-X"], ["B-X", "O"], ["O", "O"], ["B-X", "I-X"]],
            "references": [["O", "O"], ["B-X", "O"], ["O", "B-X"], ["B-X", "I-X"]],
        }
        n_observations = len(original["predictions"])
        calls = []
        test_case = self

        class RecordingMetric:
            def compute(self, predictions, references, **kwargs):
                if predictions and isinstance(predictions[0], str):
                    raise TypeError("received token slices instead of sentence rows")
                reconstructed = {"predictions": list(predictions), "references": list(references)}
                calls.append(reconstructed)
                for prediction, reference in zip(reconstructed["predictions"], reconstructed["references"]):
                    test_case.assertIsInstance(prediction, list)
                    test_case.assertIsInstance(reference, list)
                    test_case.assertEqual(len(prediction), len(reference))
                    test_case.assertEqual(len(prediction), 2)
                return {"accuracy": 0.5}

        self.evaluator.compute_metric(
            RecordingMetric(), original, strategy="bootstrap", n_resamples=9, random_state=42
        )
        self.assertGreater(len(calls), n_observations)
        call_lengths = [len(call["predictions"]) for call in calls]
        self.assertIn(n_observations, call_lengths)
        self.assertIn(n_observations - 1, call_lengths)

        def sampled_indices(call):
            return [
                next(index for index, row in enumerate(original["predictions"]) if prediction is row)
                for prediction in call["predictions"]
            ]

        leave_one_out = set()
        for call in calls:
            indices = sampled_indices(call)
            self.assertEqual(len(call["predictions"]), len(call["references"]))
            for prediction, reference, index in zip(call["predictions"], call["references"], indices):
                self.assertIs(prediction, original["predictions"][index])
                self.assertIs(reference, original["references"][index])
            if len(indices) == n_observations - 1:
                excluded = tuple(index for index in range(n_observations) if index not in indices)
                self.assertEqual(len(excluded), 1)
                leave_one_out.add(excluded[0])
        self.assertEqual(leave_one_out, set(range(n_observations)))
        self.assertTrue(
            any(
                len(set(map(id, call["predictions"]))) < len(call["predictions"])
                for call in calls
                if len(call["predictions"]) == n_observations
            )
        )

    def test_rare_and_one_sided_entities(self):
        rare = {key: [list(row) for row in rows] for key, rows in self.RAGGED_METRIC_INPUTS.items()}
        rare["references"][0] = ["B-Y"]
        rare["predictions"][0] = ["B-Y"]
        rare["predictions"][1] = ["B-Z", "O"]
        rare["references"][2] = ["O", "B-W", "I-W"]
        metric = self._load_local_seqeval()
        ordinary = metric.compute(**rare)
        self.assertIn("Y", ordinary)
        self.assertIn("Z", ordinary)
        self.assertIn("W", ordinary)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = self.evaluator.compute_metric(metric, rare, strategy="bootstrap", n_resamples=99, random_state=42)
        self._assert_entity_bootstrap_record(result["Y"], ordinary["Y"])
        self._assert_entity_bootstrap_record(result["Z"], ordinary["Z"])
        self._assert_entity_bootstrap_record(result["W"], ordinary["W"])
        self.assertEqual(result["Y"]["score"]["number"], ordinary["Y"]["number"])
        self.assertTrue(np.isnan(result["Y"]["confidence_interval"]["f1"]).all())
        self.assertTrue(np.isnan(result["Y"]["standard_error"]["f1"]))
        self.assertFalse(np.isnan(result["overall_f1"]["score"]))
        self._assert_finite_record_interval(result["overall_f1"])

    def test_all_o_and_constant_metric(self):
        all_o = {"predictions": [["O"]] * 4, "references": [["O"]] * 4}
        metric = self._load_local_seqeval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.evaluator.compute_metric(metric, all_o, strategy="bootstrap", n_resamples=9, random_state=42)
        self.assertEqual(result["overall_accuracy"]["score"], 1.0)
        self.assertTrue(np.isnan(result["overall_accuracy"]["confidence_interval"]).any())
        self.assertEqual(result["overall_accuracy"]["standard_error"], 0.0)
        warning_names = {warning.category.__name__ for warning in caught}
        self.assertTrue(warning_names)

        class ConstantMetric:
            def compute(self, predictions, references, **kwargs):
                return {"accuracy": 1.0}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            constant = self.evaluator.compute_metric(
                ConstantMetric(),
                {"predictions": [0, 1, 0, 1], "references": [0, 1, 0, 1]},
                strategy="bootstrap",
                n_resamples=9,
                random_state=42,
            )
        self.assertEqual(constant["accuracy"]["score"], 1.0)
        self.assertTrue(np.isnan(constant["accuracy"]["confidence_interval"]).any())
        self.assertEqual(constant["accuracy"]["standard_error"], 0.0)
        self.assertTrue(any(warning.category.__name__ == "DegenerateDataWarning" for warning in caught))

    def test_seed_repeatability_and_nondefault_parameters(self):
        metric = self._load_local_seqeval()
        first = self.evaluator.compute_metric(
            metric, self.RAGGED_METRIC_INPUTS, strategy="bootstrap", n_resamples=99, random_state=42
        )
        second = self.evaluator.compute_metric(
            metric, self.RAGGED_METRIC_INPUTS, strategy="bootstrap", n_resamples=99, random_state=42
        )
        np.testing.assert_allclose(
            first["overall_f1"]["confidence_interval"], second["overall_f1"]["confidence_interval"]
        )
        np.testing.assert_allclose(first["overall_f1"]["standard_error"], second["overall_f1"]["standard_error"])

        nondefault = self.evaluator.compute_metric(
            metric,
            self.RAGGED_METRIC_INPUTS,
            strategy="bootstrap",
            confidence_level=0.8,
            n_resamples=20,
            random_state=42,
        )
        self.assertEqual(nondefault["overall_f1"]["score"], first["overall_f1"]["score"])
        self.assertFalse(
            np.allclose(nondefault["overall_f1"]["confidence_interval"], first["overall_f1"]["confidence_interval"])
        )
        oracle_metric = self._load_local_seqeval()
        self._assert_matches_independent_bootstrap(
            nondefault["overall_f1"], oracle_metric, self.RAGGED_METRIC_INPUTS, ("overall_f1",), 0.8, 20, 42
        )

    def test_random_state_objects_and_none_forwarding(self):
        metric = self._load_local_seqeval()
        for factory in (lambda: np.random.RandomState(42), lambda: np.random.default_rng(42)):
            first = self.evaluator.compute_metric(
                metric, self.RAGGED_METRIC_INPUTS, strategy="bootstrap", n_resamples=9, random_state=factory()
            )
            second = self.evaluator.compute_metric(
                metric, self.RAGGED_METRIC_INPUTS, strategy="bootstrap", n_resamples=9, random_state=factory()
            )
            np.testing.assert_allclose(
                first["overall_f1"]["confidence_interval"], second["overall_f1"]["confidence_interval"], equal_nan=True
            )
            np.testing.assert_allclose(
                first["overall_f1"]["standard_error"], second["overall_f1"]["standard_error"], equal_nan=True
            )

        token_module = importlib.import_module("evaluate.evaluator.token_classification")
        with mock.patch.object(token_module, "bootstrap", wraps=bootstrap) as mocked:
            self.evaluator.compute_metric(
                metric, self.RAGGED_METRIC_INPUTS, strategy="bootstrap", n_resamples=9, random_state=None
            )
        kwargs = mocked.call_args.kwargs
        self.assertIsNone(kwargs.get("random_state"))
        self.assertNotIn("rng", kwargs)
        self.assertTrue(kwargs["paired"])
        self.assertFalse(kwargs["vectorized"])
        np.testing.assert_array_equal(kwargs["data"][0], np.arange(len(self.RAGGED_METRIC_INPUTS["references"])))

        with mock.patch.object(token_module, "bootstrap", wraps=bootstrap) as mocked:
            self.evaluator.compute_metric(
                metric, self.RAGGED_METRIC_INPUTS, strategy="bootstrap", n_resamples=9, random_state=42
            )
        self.assertEqual(mocked.call_args.kwargs["random_state"], 42)
        self.assertNotIn("rng", mocked.call_args.kwargs)

    def test_flat_scalar_custom_metric(self):
        class FlatAccuracy:
            def compute(self, predictions, references, **kwargs):
                return {
                    "accuracy": sum(prediction == reference for prediction, reference in zip(predictions, references))
                    / len(references)
                }

        metric_inputs = {"predictions": [0, 1, 0, 1, 1, 1, 0, 0], "references": [0, 1, 1, 1, 0, 1, 0, 1]}
        ordinary = FlatAccuracy().compute(**metric_inputs)
        result = self.evaluator.compute_metric(
            FlatAccuracy(), metric_inputs, strategy="bootstrap", n_resamples=99, random_state=42
        )
        self._assert_scalar_bootstrap_record(result["accuracy"], ordinary["accuracy"])
        self._assert_matches_independent_bootstrap(
            result["accuracy"], FlatAccuracy(), metric_inputs, ("accuracy",), 0.95, 99, 42
        )

    def test_metric_kwargs_reach_every_call(self):
        class KeywordMetric:
            def __init__(self):
                self.calls = []

            def compute(self, predictions, references, **kwargs):
                self.calls.append(dict(kwargs))
                return {"accuracy": 0.5}

        configured = TokenClassificationEvaluator()
        configured.METRIC_KWARGS = {"scale": 2, "suffix": True}
        metric = KeywordMetric()
        configured.compute_metric(
            metric,
            {"predictions": [1, 0, 0, 1], "references": [1, 1, 0, 1]},
            strategy="bootstrap",
            n_resamples=9,
            random_state=42,
        )
        self.assertGreater(len(metric.calls), 1)
        self.assertTrue(all(call == {"scale": 2, "suffix": True} for call in metric.calls))

    def test_seqeval_suffix_strict_and_zero_division(self):
        suffix_inputs = {
            key: [[label if label == "O" else label[2:] + "-" + label[0] for label in row] for row in rows]
            for key, rows in self.RAGGED_METRIC_INPUTS.items()
        }
        configured = TokenClassificationEvaluator()
        configured.METRIC_KWARGS = {"suffix": True, "zero_division": 0}
        metric = self._load_local_seqeval()
        ordinary = metric.compute(**suffix_inputs, **configured.METRIC_KWARGS)
        result = configured.compute_metric(
            metric, suffix_inputs, strategy="bootstrap", n_resamples=99, random_state=42
        )
        self._assert_bootstrap_result_matches_ordinary(result, ordinary)
        oracle_metric = self._load_local_seqeval()
        self._assert_matches_independent_bootstrap(
            {
                "confidence_interval": result["X"]["confidence_interval"]["f1"],
                "standard_error": result["X"]["standard_error"]["f1"],
            },
            oracle_metric,
            suffix_inputs,
            ("X", "f1"),
            0.95,
            99,
            42,
            metric_kwargs=configured.METRIC_KWARGS,
        )

        strict = TokenClassificationEvaluator()
        strict.METRIC_KWARGS = {"scheme": "IOB2", "mode": "strict", "zero_division": 0}
        strict_metric = self._load_local_seqeval()
        strict_ordinary = strict_metric.compute(**self.RAGGED_METRIC_INPUTS, **strict.METRIC_KWARGS)
        strict_result = strict.compute_metric(
            strict_metric, self.RAGGED_METRIC_INPUTS, strategy="bootstrap", n_resamples=9, random_state=42
        )
        self._assert_bootstrap_result_matches_ordinary(strict_result, strict_ordinary)

        zero = TokenClassificationEvaluator()
        zero.METRIC_KWARGS = {"zero_division": 1}
        zero_metric = self._load_local_seqeval()
        all_o = {"predictions": [["O"]] * 4, "references": [["O"]] * 4}
        zero_ordinary = zero_metric.compute(**all_o, **zero.METRIC_KWARGS)
        zero_result = zero.compute_metric(zero_metric, all_o, strategy="bootstrap", n_resamples=9, random_state=42)
        self.assertEqual(zero_result["overall_f1"]["score"], zero_ordinary["overall_f1"])

    def test_unsupported_nested_schema_and_missing_aggregate_fail(self):
        class UnsupportedNested:
            def compute(self, predictions, references, **kwargs):
                return {"entity": {"not_seqeval": 1.0}}

        with self.assertRaises(TypeError):
            self.evaluator.compute_metric(
                UnsupportedNested(),
                {"predictions": [["O"], ["B-X"]], "references": [["O"], ["B-X"]]},
                strategy="bootstrap",
                n_resamples=9,
                random_state=42,
            )

        class MissingAggregate:
            def __init__(self):
                self.calls = 0

            def compute(self, predictions, references, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return {"overall_f1": 1.0}
                return {}

        with self.assertRaises(KeyError):
            self.evaluator.compute_metric(
                MissingAggregate(),
                {"predictions": [0, 1, 0, 1], "references": [0, 1, 0, 1]},
                strategy="bootstrap",
                n_resamples=9,
                random_state=42,
            )


class TestTextGenerationEvaluator(TestCase):
    def setUp(self):
        self.data = Dataset.from_dict({"text": ["Lorem ipsum"]})
        self.pipe = DummyTextGenerationPipeline(num_return_sequences=4)
        self.evaluator = evaluator("text-generation")

    def test_class_init(self):
        evaluator = TextGenerationEvaluator()
        self.assertEqual(evaluator.task, "text-generation")
        self.assertIsNone(evaluator.default_metric_name)

        results = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="word_count",
        )
        self.assertIsInstance(results["unique_words"], int)

    @slow
    def test_default_pipe_init(self):
        results = self.evaluator.compute(data=self.data)
        self.assertIsInstance(results["unique_words"], int)

    def test_overwrite_default_metric(self):
        word_length = load("word_length")
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=word_length,
        )
        self.assertIsInstance(results["average_word_length"], int)
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="word_length",
        )
        self.assertIsInstance(results["average_word_length"], int)

    def test_process_predictions_multiple_return_sequences(self):
        processed_predictions = self.evaluator.predictions_processor(
            [
                [{"generated_text": "A"}, {"generated_text": "B"}],
                [{"generated_text": "C"}, {"generated_text": "D"}],
            ]
        )
        self.assertEqual(processed_predictions, {"data": ["A", "B", "C", "D"]})


class TestText2TextGenerationEvaluator(TestCase):
    def setUp(self):
        self.data = Dataset.from_dict(
            {
                "text": ["Lorem ipsum"] * 4,
                "label": ["Ipsum Lorem"] * 4,
            }
        )
        self.pipe = DummyText2TextGenerationPipeline()
        self.evaluator = evaluator("text2text-generation")

    def test_pipe_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
        )
        self.assertEqual(results["bleu"], 0)

    def test_class_init(self):
        evaluator = Text2TextGenerationEvaluator()
        self.assertEqual(evaluator.task, "text2text-generation")
        self.assertIsNone(evaluator.default_metric_name)

        results = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="bleu",
        )
        self.assertEqual(results["bleu"], 0)

    @slow
    def test_default_pipe_init(self):
        results = self.evaluator.compute(data=self.data)
        self.assertEqual(results["bleu"], 0)

    def test_overwrite_default_metric(self):
        rouge = load("rouge")
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=rouge,
        )
        self.assertEqual(results["rouge1"], 1.0)
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="rouge",
        )
        self.assertEqual(results["rouge1"], 1.0)

    def test_summarization(self):
        pipe = DummyText2TextGenerationPipeline(task="summarization", prefix="summary")
        e = evaluator("summarization")

        results = e.compute(
            model_or_pipeline=pipe,
            data=self.data,
        )
        self.assertEqual(results["rouge1"], 1.0)

    def test_translation(self):
        pipe = DummyText2TextGenerationPipeline(task="translation", prefix="translation")
        e = evaluator("translation")

        results = e.compute(
            model_or_pipeline=pipe,
            data=self.data,
        )
        self.assertEqual(results["bleu"], 0)


class TestAutomaticSpeechRecognitionEvaluator(TestCase):
    def setUp(self):
        self.data = Dataset.from_dict(
            {
                "path": [
                    # Examples copied from default speech model of
                    # `automic-speech-recognition` pipeline:
                    # https://huggingface.co/facebook/wav2vec2-base-960h
                    # https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/__init__.py#L161
                    "https://cdn-media.huggingface.co/speech_samples/sample1.flac",
                    "https://cdn-media.huggingface.co/speech_samples/sample2.flac",
                ],
                "sentence": ["Ipsum Lorem"] * 2,
            }
        )
        self.pipe = DummyAutomaticSpeechRecognitionPipeline()
        self.evaluator = evaluator("automatic-speech-recognition")

    def test_pipe_init(self):
        print(self.evaluator)
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
        )
        print(results)
        self.assertEqual(results["wer"], 1.0)

    def test_class_init(self):
        evaluator = AutomaticSpeechRecognitionEvaluator()
        self.assertEqual(evaluator.task, "automatic-speech-recognition")
        self.assertIsNone(evaluator.default_metric_name)

        results = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="wer",
        )
        self.assertEqual(results["wer"], 1.0)

    @slow
    def test_default_pipe_init(self):
        results = self.evaluator.compute(data=self.data)
        self.assertGreater(results["wer"], 1.0)

    def test_overwrite_default_metric(self):
        cer = load("cer")
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=cer,
        )
        self.assertEqual(results["cer"], 0.7272727272727273)

        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="cer",
        )
        self.assertEqual(results["cer"], 0.7272727272727273)


class TestAudioClassificationEvaluator(TestCase):
    def setUp(self):
        self.data = Dataset.from_dict(
            {"file": ["https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"], "label": [11]}
        )
        self.raw_data = Dataset.from_dict(
            {
                "audio": [
                    np.array(
                        [-0.00048828, -0.00018311, -0.00137329, 0.00079346, 0.00091553, 0.00085449], dtype=np.float32
                    )
                ],
                "label": [11],
            }
        )
        self.default_model = "superb/wav2vec2-base-superb-ks"
        self.pipe = DummyAudioClassificationPipeline()
        self.evaluator = evaluator("audio-classification")
        self.label_mapping = AutoConfig.from_pretrained(self.default_model).label2id

    def test_pipe_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

    def test_raw_pipe_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe, data=self.raw_data, label_mapping=self.label_mapping, input_column="audio"
        )
        self.assertEqual(results["accuracy"], 0)

    @slow
    def test_model_init(self):
        results = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="accuracy",
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

        model = AutoModelForAudioClassification.from_pretrained(self.default_model)
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.default_model)

        results = self.evaluator.compute(
            model_or_pipeline=model,
            data=self.data,
            metric="accuracy",
            feature_extractor=feature_extractor,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

    def test_class_init(self):
        evaluator = AudioClassificationEvaluator()
        self.assertEqual(evaluator.task, "audio-classification")
        self.assertIsNone(evaluator.default_metric_name)

        results = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
            label_mapping=self.label_mapping,
        )
        results_raw = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.raw_data,
            label_mapping=self.label_mapping,
            metric="accuracy",
            input_column="audio",
        )
        self.assertEqual(results_raw["accuracy"], 0)
        self.assertEqual(results["accuracy"], 0)

    @slow
    def test_default_pipe_init(self):
        results = self.evaluator.compute(
            data=self.data,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

    def test_overwrite_default_metric(self):
        accuracy = load("accuracy")
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=accuracy,
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)
        results = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="accuracy",
            label_mapping=self.label_mapping,
        )
        self.assertEqual(results["accuracy"], 0)

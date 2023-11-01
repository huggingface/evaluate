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
from unittest import TestCase, mock

import numpy as np
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from PIL import Image
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
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][0], 0.33333, 5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][1], 0.666666, 5)
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
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][0], 0.333333, 5)
        self.assertAlmostEqual(results["accuracy"]["confidence_interval"][1], 0.666666, 5)
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

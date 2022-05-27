# Copyright 2020 HuggingFace Inc.
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

import doctest
import glob
import importlib
import inspect
import os
import re
from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch

import numpy as np
import pytest
from absl.testing import parameterized

import evaluate
from evaluate import load

from .utils import for_all_test_methods, local, slow


REQUIRE_FAIRSEQ = {"comet"}
_has_fairseq = importlib.util.find_spec("fairseq") is not None

UNSUPPORTED_ON_WINDOWS = {"code_eval"}
_on_windows = os.name == "nt"


def skip_if_metric_requires_fairseq(test_case):
    @wraps(test_case)
    def wrapper(self, evaluation_module_name, evaluation_module_type):
        if not _has_fairseq and evaluation_module_name in REQUIRE_FAIRSEQ:
            self.skipTest('"test requires Fairseq"')
        else:
            test_case(self, evaluation_module_name, evaluation_module_type)

    return wrapper


def skip_on_windows_if_not_windows_compatible(test_case):
    @wraps(test_case)
    def wrapper(self, evaluation_module_name, evaluation_module_type):
        if _on_windows and evaluation_module_name in UNSUPPORTED_ON_WINDOWS:
            self.skipTest('"test not supported on Windows"')
        else:
            test_case(self, evaluation_module_name, evaluation_module_type)

    return wrapper


def get_local_module_names():
    metrics = [metric_dir.split(os.sep)[-2] for metric_dir in glob.glob("./metrics/*/")]
    comparisons = [metric_dir.split(os.sep)[-2] for metric_dir in glob.glob("./comparisons/*/")]
    measurements = [metric_dir.split(os.sep)[-2] for metric_dir in glob.glob("./measurements/*/")]

    evaluation_modules = metrics + comparisons + measurements
    evaluation_module_types = (
        ["metric"] * len(metrics) + ["comparison"] * len(comparisons) + ["measurement"] * len(measurements)
    )

    return [
        {"testcase_name": f"{t}_{x}", "evaluation_module_name": x, "evaluation_module_type": t}
        for x, t in zip(evaluation_modules, evaluation_module_types)
        if x != "gleu"  # gleu is unfinished
    ]


@parameterized.named_parameters(get_local_module_names())
@for_all_test_methods(skip_if_metric_requires_fairseq, skip_on_windows_if_not_windows_compatible)
@local
class LocalModuleTest(parameterized.TestCase):
    INTENSIVE_CALLS_PATCHER = {}
    evaluation_module_name = None
    evaluation_module_type = None

    def test_load(self, evaluation_module_name, evaluation_module_type):
        doctest.ELLIPSIS_MARKER = "[...]"
        evaluation_module = importlib.import_module(
            evaluate.loading.evaluation_module_factory(
                os.path.join(evaluation_module_type + "s", evaluation_module_name), module_type=type
            ).module_path
        )
        evaluation_instance = evaluate.loading.import_main_class(evaluation_module.__name__)
        # check parameters
        parameters = inspect.signature(evaluation_instance._compute).parameters
        self.assertTrue(all([p.kind != p.VAR_KEYWORD for p in parameters.values()]))  # no **kwargs
        # run doctest
        with self.patch_intensive_calls(evaluation_module_name, evaluation_module.__name__):
            with self.use_local_metrics(evaluation_module_type):
                try:
                    results = doctest.testmod(evaluation_module, verbose=True, raise_on_error=True)
                except doctest.UnexpectedException as e:
                    raise e.exc_info[1]  # raise the exception that doctest caught
        self.assertEqual(results.failed, 0)
        self.assertGreater(results.attempted, 1)

    @slow
    def test_load_real_metric(self, evaluation_module_name, evaluation_module_type):
        doctest.ELLIPSIS_MARKER = "[...]"
        metric_module = importlib.import_module(
            evaluate.loading.evaluation_module_factory(
                os.path.join(evaluation_module_type, evaluation_module_name)
            ).module_path
        )
        # run doctest
        with self.use_local_metrics():
            results = doctest.testmod(metric_module, verbose=True, raise_on_error=True)
        self.assertEqual(results.failed, 0)
        self.assertGreater(results.attempted, 1)

    @contextmanager
    def patch_intensive_calls(self, evaluation_module_name, module_name):
        if evaluation_module_name in self.INTENSIVE_CALLS_PATCHER:
            with self.INTENSIVE_CALLS_PATCHER[evaluation_module_name](module_name):
                yield
        else:
            yield

    @contextmanager
    def use_local_metrics(self, evaluation_module_type):
        def load_local_metric(evaluation_module_name, *args, **kwargs):
            return load(os.path.join(evaluation_module_type + "s", evaluation_module_name), *args, **kwargs)

        with patch("evaluate.load") as mock_load:
            mock_load.side_effect = load_local_metric
            yield

    @classmethod
    def register_intensive_calls_patcher(cls, evaluation_module_name):
        def wrapper(patcher):
            patcher = contextmanager(patcher)
            cls.INTENSIVE_CALLS_PATCHER[evaluation_module_name] = patcher
            return patcher

        return wrapper


# Metrics intensive calls patchers
# --------------------------------


@LocalModuleTest.register_intensive_calls_patcher("bleurt")
def patch_bleurt(module_name):
    import tensorflow.compat.v1 as tf
    from bleurt.score import Predictor

    tf.flags.DEFINE_string("sv", "", "")  # handle pytest cli flags

    class MockedPredictor(Predictor):
        def predict(self, input_dict):
            assert len(input_dict["input_ids"]) == 2
            return np.array([1.03, 1.04])

    # mock predict_fn which is supposed to do a forward pass with a bleurt model
    with patch("bleurt.score._create_predictor") as mock_create_predictor:
        mock_create_predictor.return_value = MockedPredictor()
        yield


@LocalModuleTest.register_intensive_calls_patcher("bertscore")
def patch_bertscore(module_name):
    import torch

    def bert_cos_score_idf(model, refs, *args, **kwargs):
        return torch.tensor([[1.0, 1.0, 1.0]] * len(refs))

    # mock get_model which is supposed to do download a bert model
    # mock bert_cos_score_idf which is supposed to do a forward pass with a bert model
    with patch("bert_score.scorer.get_model"), patch(
        "bert_score.scorer.bert_cos_score_idf"
    ) as mock_bert_cos_score_idf:
        mock_bert_cos_score_idf.side_effect = bert_cos_score_idf
        yield


@LocalModuleTest.register_intensive_calls_patcher("comet")
def patch_comet(module_name):
    def load_from_checkpoint(model_path):
        class Model:
            def predict(self, data, *args, **kwargs):
                assert len(data) == 2
                scores = [0.19, 0.92]
                return scores, sum(scores) / len(scores)

        return Model()

    # mock load_from_checkpoint which is supposed to do download a bert model
    # mock load_from_checkpoint which is supposed to do download a bert model
    with patch("comet.download_model") as mock_download_model:
        mock_download_model.return_value = None
        with patch("comet.load_from_checkpoint") as mock_load_from_checkpoint:
            mock_load_from_checkpoint.side_effect = load_from_checkpoint
            yield


def test_seqeval_raises_when_incorrect_scheme():
    metric = load(os.path.join("metrics", "seqeval"))
    wrong_scheme = "ERROR"
    error_message = f"Scheme should be one of [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU], got {wrong_scheme}"
    with pytest.raises(ValueError, match=re.escape(error_message)):
        metric.compute(predictions=[], references=[], scheme=wrong_scheme)

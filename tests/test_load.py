import importlib
import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pytest
from datasets import DownloadConfig

import evaluate
from evaluate.loading import (
    CachedEvaluationModuleFactory,
    HubEvaluationModuleFactory,
    LocalEvaluationModuleFactory,
    evaluation_module_factory,
)

from .utils import OfflineSimulationMode, offline


SAMPLE_METRIC_IDENTIFIER = "lvwerra/test"

METRIC_LOADING_SCRIPT_NAME = "__dummy_metric1__"

METRIC_LOADING_SCRIPT_CODE = """
import evaluate
from evaluate import EvaluationModuleInfo
from datasets import Features, Value

class __DummyMetric1__(evaluate.EvaluationModule):

    def _info(self):
        return EvaluationModuleInfo(features=Features({"predictions": Value("int"), "references": Value("int")}))

    def _compute(self, predictions, references):
        return {"__dummy_metric1__": sum(int(p == r) for p, r in zip(predictions, references))}
"""


@pytest.fixture
def metric_loading_script_dir(tmp_path):
    script_name = METRIC_LOADING_SCRIPT_NAME
    script_dir = tmp_path / script_name
    script_dir.mkdir()
    script_path = script_dir / f"{script_name}.py"
    with open(script_path, "w") as f:
        f.write(METRIC_LOADING_SCRIPT_CODE)
    return str(script_dir)


class ModuleFactoryTest(TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, metric_loading_script_dir):
        self._metric_loading_script_dir = metric_loading_script_dir

    def setUp(self):
        self.hf_modules_cache = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()
        self.download_config = DownloadConfig(cache_dir=self.cache_dir)
        self.dynamic_modules_path = evaluate.loading.init_dynamic_modules(
            name="test_datasets_modules_" + os.path.basename(self.hf_modules_cache),
            hf_modules_cache=self.hf_modules_cache,
        )

    def test_HubEvaluationModuleFactory_with_internal_import(self):
        # "squad_v2" requires additional imports (internal)
        factory = HubEvaluationModuleFactory(
            "evaluate-metric/squad_v2",
            module_type="metric",
            download_config=self.download_config,
            dynamic_modules_path=self.dynamic_modules_path,
        )
        module_factory_result = factory.get_module()
        assert importlib.import_module(module_factory_result.module_path) is not None

    def test_HubEvaluationModuleFactory_with_external_import(self):
        # "bleu" requires additional imports (external from github)
        factory = HubEvaluationModuleFactory(
            "evaluate-metric/bleu",
            module_type="metric",
            download_config=self.download_config,
            dynamic_modules_path=self.dynamic_modules_path,
        )
        module_factory_result = factory.get_module()
        assert importlib.import_module(module_factory_result.module_path) is not None

    def test_HubEvaluationModuleFactoryWithScript(self):
        factory = HubEvaluationModuleFactory(
            SAMPLE_METRIC_IDENTIFIER,
            download_config=self.download_config,
            dynamic_modules_path=self.dynamic_modules_path,
        )
        module_factory_result = factory.get_module()
        assert importlib.import_module(module_factory_result.module_path) is not None

    def test_LocalMetricModuleFactory(self):
        path = os.path.join(self._metric_loading_script_dir, f"{METRIC_LOADING_SCRIPT_NAME}.py")
        factory = LocalEvaluationModuleFactory(
            path, download_config=self.download_config, dynamic_modules_path=self.dynamic_modules_path
        )
        module_factory_result = factory.get_module()
        assert importlib.import_module(module_factory_result.module_path) is not None

    def test_CachedMetricModuleFactory(self):
        path = os.path.join(self._metric_loading_script_dir, f"{METRIC_LOADING_SCRIPT_NAME}.py")
        factory = LocalEvaluationModuleFactory(
            path, download_config=self.download_config, dynamic_modules_path=self.dynamic_modules_path
        )
        module_factory_result = factory.get_module()
        for offline_mode in OfflineSimulationMode:
            with offline(offline_mode):
                factory = CachedEvaluationModuleFactory(
                    METRIC_LOADING_SCRIPT_NAME,
                    dynamic_modules_path=self.dynamic_modules_path,
                )
                module_factory_result = factory.get_module()
                assert importlib.import_module(module_factory_result.module_path) is not None

    def test_cache_with_remote_canonical_module(self):
        metric = "accuracy"
        evaluation_module_factory(
            metric, download_config=self.download_config, dynamic_modules_path=self.dynamic_modules_path
        )

        for offline_mode in OfflineSimulationMode:
            with offline(offline_mode):
                evaluation_module_factory(
                    metric, download_config=self.download_config, dynamic_modules_path=self.dynamic_modules_path
                )

    def test_cache_with_remote_community_module(self):
        metric = "lvwerra/test"
        evaluation_module_factory(
            metric, download_config=self.download_config, dynamic_modules_path=self.dynamic_modules_path
        )

        for offline_mode in OfflineSimulationMode:
            with offline(offline_mode):
                evaluation_module_factory(
                    metric, download_config=self.download_config, dynamic_modules_path=self.dynamic_modules_path
                )


class FilesToHashTest(TestCase):
    _HASHTAG_METRIC_TEMPLATE = """import evaluate
from datasets import Features, Value

class HashtagCount(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description="Count a hashtag",
            citation="",
            features=Features({"predictions": Value("string")}),
        )

    def _compute(self, predictions):
        return {"count": sum(text.count("HASHTAG") for text in predictions)}
"""

    def test_files_to_hash_distinguishes_hashtag_string_literals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.py"
            second = Path(tmpdir) / "second.py"
            first.write_text(self._HASHTAG_METRIC_TEMPLATE.replace("HASHTAG", "#positive"), encoding="utf-8")
            second.write_text(self._HASHTAG_METRIC_TEMPLATE.replace("HASHTAG", "#negative"), encoding="utf-8")

            self.assertNotEqual(
                evaluate.loading.files_to_hash([str(first)]),
                evaluate.loading.files_to_hash([str(second)]),
            )

    def test_local_metric_load_invalidates_cache_when_hashtag_literal_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as cache_dir:
            dynamic_modules_path = evaluate.loading.init_dynamic_modules(
                name="test_hashtag_hash_" + os.path.basename(cache_dir),
                hf_modules_cache=cache_dir,
            )
            script = Path(tmpdir) / "hashtag_count.py"
            predictions = ["#positive", "#positive", "#negative"]

            with patch("evaluate.loading.init_dynamic_modules", return_value=dynamic_modules_path):
                script.write_text(self._HASHTAG_METRIC_TEMPLATE.replace("HASHTAG", "#positive"), encoding="utf-8")
                first = evaluate.load(str(script), keep_in_memory=True).compute(predictions=predictions)

                script.write_text(self._HASHTAG_METRIC_TEMPLATE.replace("HASHTAG", "#negative"), encoding="utf-8")
                second = evaluate.load(str(script), keep_in_memory=True).compute(predictions=predictions)

            self.assertEqual(first, {"count": 2})
            self.assertEqual(second, {"count": 1})

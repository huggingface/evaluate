import importlib
import os
import tempfile
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

    def test_evaluation_module_factory_local_py_path_passes_download_config(self):
        # Regression test for https://github.com/huggingface/evaluate/issues/709:
        # evaluation_module_factory must forward download_config to LocalEvaluationModuleFactory
        # when a direct .py path is given (path.endswith(filename) branch).
        py_path = os.path.join(self._metric_loading_script_dir, f"{METRIC_LOADING_SCRIPT_NAME}.py")
        with patch("evaluate.loading.LocalEvaluationModuleFactory", wraps=LocalEvaluationModuleFactory) as spy:
            evaluation_module_factory(
                py_path,
                download_config=self.download_config,
                dynamic_modules_path=self.dynamic_modules_path,
            )
            spy.assert_called_once()
            _, kwargs = spy.call_args
            assert kwargs.get("download_config") is self.download_config

    def test_evaluation_module_factory_local_dir_path_passes_download_config(self):
        # Regression test for https://github.com/huggingface/evaluate/issues/709:
        # evaluation_module_factory must forward download_config to LocalEvaluationModuleFactory
        # when a directory path is given (combined_path branch).
        with patch("evaluate.loading.LocalEvaluationModuleFactory", wraps=LocalEvaluationModuleFactory) as spy:
            evaluation_module_factory(
                self._metric_loading_script_dir,
                download_config=self.download_config,
                dynamic_modules_path=self.dynamic_modules_path,
            )
            spy.assert_called_once()
            _, kwargs = spy.call_args
            assert kwargs.get("download_config") is self.download_config

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

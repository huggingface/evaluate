import importlib
import os
import tempfile
from unittest import TestCase

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


def test_copy_script_metadata_path_when_ancestor_dir_contains_py():
    """Regression: meta_path derivation when an ancestor directory contains ".py".

    The previous implementation used `importable_local_file.split(".py")[0]`,
    which splits on every occurrence, so a cache path under ~/.pyenv/... wrote
    the metadata to ~/.json — outside the cache tree — instead of next to the
    copied script.
    """
    import json

    from evaluate.loading import _copy_script_and_other_resources_in_importable_dir

    with tempfile.TemporaryDirectory() as root:
        # Simulate a pyenv-style cache path: ancestor directory contains ".py".
        importable_directory_path = os.path.join(root, ".pyenv", "evaluate_modules")
        # The FileLock acquired inside the function needs the parent dir to exist.
        os.makedirs(os.path.dirname(importable_directory_path), exist_ok=True)
        subdirectory_name = "abcd1234"
        name = "accuracy"

        original_script_path = os.path.join(root, "src_accuracy.py")
        with open(original_script_path, "w", encoding="utf-8") as f:
            f.write("# dummy metric script\n")

        _copy_script_and_other_resources_in_importable_dir(
            name=name,
            importable_directory_path=importable_directory_path,
            subdirectory_name=subdirectory_name,
            original_local_path=original_script_path,
            local_imports=[],
            additional_files=[],
            download_mode=None,
        )

        expected_meta = os.path.join(importable_directory_path, subdirectory_name, name + ".json")
        leaked_meta = os.path.join(root, ".json")

        assert os.path.exists(expected_meta), f"meta file missing at {expected_meta}"
        assert not os.path.exists(leaked_meta), f"meta file leaked to {leaked_meta}"
        with open(expected_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["original file path"] == original_script_path

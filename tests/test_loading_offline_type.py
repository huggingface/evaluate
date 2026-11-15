from pathlib import Path
from unittest import mock

import pytest

import evaluate
from evaluate.loading import HubEvaluationModuleFactory, init_dynamic_modules


@pytest.mark.parametrize(
    "cached_types,requested_type,expected_type",
    [
        (("metric", "comparison"), "comparison", "comparison"),
        (("metric",), "comparison", None),
        (("metric", "comparison"), None, "metric"),
        (("comparison",), None, "comparison"),
    ],
)
def test_offline_cache_honors_requested_module_type(
    tmp_path, monkeypatch, cached_types, requested_type, expected_type
):
    cache = init_dynamic_modules(name=f"test_offline_types_{tmp_path.name}", hf_modules_cache=tmp_path)
    monkeypatch.setattr("evaluate.loading.init_dynamic_modules", lambda: cache)
    monkeypatch.setattr(evaluate.config, "HF_EVALUATE_OFFLINE", True)
    for module_type in cached_types:
        directory = "metrics" if module_type == "metric" else "comparisons"
        script = Path(__file__).parents[1] / directory / "exact_match" / "exact_match.py"
        factory = HubEvaluationModuleFactory(f"evaluate-{module_type}/exact_match")
        with mock.patch.object(factory, "download_loading_script", return_value=str(script)):
            factory.get_module()

    if expected_type is None:
        with pytest.raises(FileNotFoundError):
            evaluate.load("exact_match", module_type=requested_type)
    else:
        module = evaluate.load("exact_match", module_type=requested_type)
        assert module.module_type == expected_type

import pytest

import evaluate


@pytest.mark.parametrize("base_class", ["EvaluationModule", "Metric", "Comparison", "Measurement"])
@pytest.mark.parametrize("import_implementation", [False, True])
def test_local_module_can_import_base_class_directly(tmp_path, base_class, import_implementation):
    script = tmp_path / "local_count.py"
    source = f"""from evaluate import {base_class}, {base_class}Info
from datasets import Features, Value

class LocalCount({base_class}):
    def _info(self):
        return {base_class}Info(
            description="Local count", citation="",
            features=Features({{"predictions": Value("int64")}}))

    def _compute(self, predictions):
        return {{"count": len(predictions)}}
"""
    if import_implementation:
        (tmp_path / "implementation.py").write_text(source, encoding="utf-8")
        script.write_text("from .implementation import LocalCount\n", encoding="utf-8")
    else:
        script.write_text(source, encoding="utf-8")
    metric = evaluate.load(str(script), keep_in_memory=True)
    assert metric.__class__.__name__ == "LocalCount"
    assert metric.compute(predictions=[0, 1, 2]) == {"count": 3}

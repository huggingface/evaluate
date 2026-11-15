import json

import numpy as np
import pytest

import evaluate


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_save_serialization_error_preserves_destination(tmp_path, existing, nested):
    destination = tmp_path / "results.json"
    if existing:
        evaluate.save(destination, accuracy=0.75)
        original = destination.read_bytes()
    result = {"accuracy": 1.0, "details": {"scores": np.array([1, 2])}} if nested else {"score": np.int64(1)}

    with pytest.raises(TypeError, match="JSON serializable"):
        evaluate.save(destination, **result)

    if existing:
        assert destination.read_bytes() == original
        assert json.loads(destination.read_text(encoding="utf-8"))["accuracy"] == 0.75
    else:
        assert not destination.exists()


def test_save_valid_result_can_replace_existing_destination(tmp_path):
    destination = tmp_path / "results.json"
    evaluate.save(destination, accuracy=0.75)
    assert evaluate.save(destination, accuracy=1.0) == destination
    assert json.loads(destination.read_text(encoding="utf-8"))["accuracy"] == 1.0

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def exact_match_module():
    metric_path = Path(__file__).parents[1] / "metrics" / "exact_match" / "exact_match.py"
    spec = importlib.util.spec_from_file_location("exact_match", metric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exact_match(exact_match_module):
    return exact_match_module.ExactMatch()


@pytest.mark.parametrize(
    "prediction, reference, kwargs",
    [
        ("42", "17", {"ignore_numbers": True}),
        ("?", "!", {"ignore_punctuation": True}),
        ("42!", "17?", {"ignore_numbers": True, "ignore_punctuation": True}),
        ("42", "17", {"regexes_to_ignore": [r".+"]}),
    ],
)
def test_different_inputs_erased_by_normalization_do_not_match(exact_match, prediction, reference, kwargs):
    result = exact_match.compute(predictions=[prediction], references=[reference], **kwargs)

    assert result == {"exact_match": 0.0}


@pytest.mark.parametrize(
    "prediction, reference, kwargs, expected",
    [
        ("42", "17", {}, 0.0),
        ("42", "42", {}, 1.0),
        ("Paris", "paris", {"ignore_case": True}, 1.0),
        ("Chapter 3", "Chapter 4", {"ignore_numbers": True}, 1.0),
        ("Answer: 42", "Answer: 17", {"regexes_to_ignore": [r"\d+"]}, 1.0),
        ("42", "42", {"ignore_numbers": True}, 1.0),
        ("", "", {"ignore_punctuation": True}, 1.0),
    ],
)
def test_exact_match_normalization_preserves_intended_matches(exact_match, prediction, reference, kwargs, expected):
    result = exact_match.compute(predictions=[prediction], references=[reference], **kwargs)

    assert result == {"exact_match": expected}

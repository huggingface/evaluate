from pathlib import Path

import pytest
from packaging.version import parse

import evaluate


@pytest.mark.parametrize(
    ("directory", "module_type"), [("metrics/meteor", "metric"), ("measurements/word_length", "measurement")]
)
@pytest.mark.parametrize("installed", [True, False])
def test_load_nltk_modules_only_downloads_missing_data(monkeypatch, directory, module_type, installed):
    nltk = pytest.importorskip("nltk")
    nltk_version = parse(nltk.__version__)
    if nltk_version < parse("3.6.6"):
        pytest.skip("Both Punkt and the multilingual WordNet data are exercised on current NLTK")
    downloads = []

    def find(resource):
        if not installed:
            raise LookupError(resource)

    def tokenize(text):
        find("punkt")
        return []

    monkeypatch.setattr(nltk.data, "find", find)
    monkeypatch.setattr(nltk, "word_tokenize", tokenize)
    monkeypatch.setattr(nltk, "download", downloads.append)
    path = Path(__file__).resolve().parents[1] / directory
    evaluate.load(str(path), module_type=module_type, keep_in_memory=True)

    punkt = "punkt_tab" if nltk_version >= parse("3.9.0") else "punkt"
    required = ["wordnet", punkt, "omw-1.4"] if module_type == "metric" else [punkt]
    assert downloads == ([] if installed else required)

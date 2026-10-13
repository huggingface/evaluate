import json
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import evaluate


result_dict = {"metric": 1.0, "model_name": "x"}

SAVE_EXTRA_KEYS = ["_timestamp", "_git_commit_hash", "_evaluate_version", "_python_version", "_interpreter_path"]


class TestSave(TestCase):
    def setUp(self):
        self.save_path = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.save_path)

    def test_save_to_folder(self):
        file_path = evaluate.save(self.save_path, **result_dict)
        with open(file_path, "r") as f:
            loaded_result_dict = json.load(f)
        for key in SAVE_EXTRA_KEYS:
            _ = loaded_result_dict.pop(key)
        self.assertDictEqual(result_dict, loaded_result_dict)

    def test_save_to_folder_nested(self):
        file_path = evaluate.save(self.save_path / "sub_dir1/sub_dir2", **result_dict)
        with open(file_path, "r") as f:
            loaded_result_dict = json.load(f)
        for key in SAVE_EXTRA_KEYS:
            _ = loaded_result_dict.pop(key)
        self.assertDictEqual(result_dict, loaded_result_dict)

    def test_save_to_file(self):
        _ = evaluate.save(self.save_path / "test.json", **result_dict)
        with open(self.save_path / "test.json", "r") as f:
            loaded_result_dict = json.load(f)
        for key in SAVE_EXTRA_KEYS:
            _ = loaded_result_dict.pop(key)
        self.assertDictEqual(result_dict, loaded_result_dict)


def test_save_without_git(tmp_path, monkeypatch):
    import evaluate.saving

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(evaluate.saving.subprocess, "run", missing_git)
    path = evaluate.save(tmp_path / "result.json", accuracy=0.75)
    data = json.loads(path.read_text())
    assert data["accuracy"] == 0.75
    assert data["_git_commit_hash"] is None

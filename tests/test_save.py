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

    def test_save_preserves_existing_file_on_serialization_error(self):
        target = self.save_path / "results.json"
        evaluate.save(target, **result_dict)
        previous_bytes = target.read_bytes()

        with self.assertRaises(TypeError):
            evaluate.save(target, accuracy=1.0, details={"scores": {1, 2}})

        self.assertEqual(target.read_bytes(), previous_bytes)

    def test_save_does_not_create_file_on_serialization_error(self):
        target = self.save_path / "nested" / "new_results.json"

        with self.assertRaises(TypeError):
            evaluate.save(target, bad=object())

        self.assertFalse(target.exists())

    def test_save_overwrites_existing_file_when_valid(self):
        target = self.save_path / "results.json"
        evaluate.save(target, accuracy=0.5)
        evaluate.save(target, accuracy=0.9)

        with open(target, "r") as f:
            loaded_result_dict = json.load(f)
        self.assertEqual(loaded_result_dict["accuracy"], 0.9)

import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest

import transformers
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoModelForImageClassification, AutoFeatureExtractor, AutoTokenizer, pipeline

from evaluate import evaluator


def onerror(func, path, exc_info):
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


class TestEvaluatorTrainerParity(unittest.TestCase):
    def setUp(self):
        self.dir_path = tempfile.mkdtemp("evaluator_trainer_parity_test")

        transformers_version = transformers.__version__
        branch = ""
        if not transformers_version.endswith(".dev0"):
            branch = f"--branch v{transformers_version}"
        subprocess.run(
            f"git clone --depth 3 --filter=blob:none --sparse {branch} https://github.com/huggingface/transformers",
            shell=True,
            cwd=self.dir_path,
        )

    def tearDown(self):
        shutil.rmtree(self.dir_path, onerror=onerror)


    def test_text_classification_parity(self):
        model_name = "philschmid/tiny-bert-sst2-distilled"

        subprocess.run(
            "git sparse-checkout set examples/pytorch/text-classification",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        subprocess.run(
            f"python examples/pytorch/text-classification/run_glue.py"
            f" --model_name_or_path {model_name}"
            f" --task_name sst2"
            f" --do_eval"
            f" --max_seq_length 9999999999"  # rely on tokenizer.model_max_length for max_length
            f" --output_dir {os.path.join(self.dir_path, 'textclassification_sst2_transformers')}"
            f" --max_eval_samples 80",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        with open(
            f"{os.path.join(self.dir_path, 'textclassification_sst2_transformers', 'eval_results.json')}", "r"
        ) as f:
            transformers_results = json.load(f)

        eval_dataset = load_dataset("glue", "sst2", split="validation[:80]")

        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer)

        e = evaluator(task="text-classification")
        evaluator_results = e.compute(
            model_or_pipeline=pipe,
            data=eval_dataset,
            metric="accuracy",
            input_column="sentence",
            label_column="label",
            label_mapping={"negative": 0, "positive": 1},
            strategy="simple",
        )

        self.assertEqual(transformers_results["eval_accuracy"], evaluator_results["accuracy"])

    def test_image_classification_parity(self):
        model_name = "nateraw/vit-base-beans"

        subprocess.run(
            "git sparse-checkout set examples/pytorch/image-classification",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        subprocess.run(
            f"python examples/pytorch/image-classification/run_image_classification.py"
            f" --model_name_or_path {model_name}"
            f" --dataset_name beans"
            f" --do_eval"
            f" --remove_unused_columns False"
            f" --seed 42"
            f" --output_dir {os.path.join(self.dir_path, 'imageclassification_beans_transformers')}"
            f" --max_eval_samples 100",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        with open(
            f"{os.path.join(self.dir_path, 'imageclassification_beans_transformers', 'eval_results.json')}", "r"
        ) as f:
            transformers_results = json.load(f)


        raw_dataset = load_dataset("beans")
        eval_dataset = raw_dataset["validation"].shuffle(seed=42).select(range(100))


        model = AutoModelForImageClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        pipe = pipeline(task="image-classification", model=model, feature_extractor=feature_extractor)

        e = evaluator(task="image-classification")
        evaluator_results = e.compute(
            model_or_pipeline=pipe,
            data=eval_dataset,
            metric="accuracy",
            input_column="image",
            label_column="labels",
            label_mapping=model.config.label2id,
            strategy="simple",
        )

        self.assertEqual(transformers_results["eval_accuracy"], evaluator_results["accuracy"])

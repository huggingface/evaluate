import json
import os
import subprocess
import unittest
from datetime import datetime

from datasets import ClassLabel, Dataset, Features, Sequence, Value, load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from evaluate import TokenClassificationEvaluator, evaluator, load


class TestTokenClassificationEvaluator(unittest.TestCase):
    def setUp(self):
        features = Features(
            {
                "tokens": Sequence(feature=Value(dtype="string")),
                "ner_tags": Sequence(feature=ClassLabel(names=["O", "FAKE", "FAKE2", "B-ORG"])),
            }
        )

        self.data = Dataset.from_dict(
            {
                "tokens": [["By", "stumps", "Kent", "had", "reached", "108", "for", "three", "."]],
                "ner_tags": [[0, 0, 3, 0, 0, 0, 0, 0, 0]],
            },
            features=features,
        )
        self.default_model = "elastic/distilbert-base-uncased-finetuned-conll03-english"
        self.input_column = "tokens"
        self.ref_column = "ner_tags"

        self.model = AutoModelForTokenClassification.from_pretrained(self.default_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.default_model)
        self.pipe = pipeline(task="token-classification", model=self.model, tokenizer=self.tokenizer)
        self.evaluator = evaluator("token-classification")

    def test_model_init(self):
        scores = self.evaluator.compute(
            model_or_pipeline=self.default_model,
            data=self.data,
            metric="seqeval",
            input_column=self.input_column,
            ref_column=self.ref_column,
        )
        self.assertEqual(scores["overall_accuracy"], 1.0)
        scores = self.evaluator.compute(
            model_or_pipeline=self.model,
            data=self.data,
            metric="seqeval",
            tokenizer=self.tokenizer,
            input_column=self.input_column,
            ref_column=self.ref_column,
        )
        self.assertEqual(scores["overall_accuracy"], 1.0)

    def test_class_init(self):
        evaluator = TokenClassificationEvaluator()
        self.assertEqual(evaluator.task, "token-classification")
        self.assertIsNone(evaluator.default_metric_name)

        scores = evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="seqeval",
            input_column=self.input_column,
            ref_column=self.ref_column,
        )
        self.assertEqual(scores["overall_accuracy"], 1.0)

    def test_default_pipe_init(self):
        scores = self.evaluator.compute(
            data=self.data,
            input_column=self.input_column,
            ref_column=self.ref_column,
        )
        self.assertEqual(scores["overall_accuracy"], 8 / 9)

    def test_overwrite_default_metric(self):
        accuracy = load("seqeval")
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric=accuracy,
            input_column=self.input_column,
            ref_column=self.ref_column,
        )
        self.assertEqual(scores["overall_accuracy"], 1.0)
        scores = self.evaluator.compute(
            model_or_pipeline=self.pipe,
            data=self.data,
            metric="seqeval",
            input_column=self.input_column,
            ref_column=self.ref_column,
        )
        self.assertEqual(scores["overall_accuracy"], 1.0)

    def test_wrong_task(self):
        self.assertRaises(KeyError, evaluator, "bad_task")

    def test_match_transformers_example(self):
        model_name = "elastic/distilbert-base-uncased-finetuned-conll03-english"

        tps = datetime.now().isoformat()
        dir_path = f"/tmp/{tps}_tokenclassification"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        subprocess.run(
            "git clone --depth 3 --filter=blob:none --sparse https://github.com/huggingface/transformers", shell=True
        )

        os.chdir("transformers")
        subprocess.run("git sparse-checkout set examples/pytorch/token-classification", shell=True)

        subprocess.run(
            f"python3 examples/pytorch/token-classification/run_ner.py"
            f" --model_name_or_path {model_name}"
            f" --dataset_name conll2003"
            f" --do_eval"
            f" --output_dir {dir_path}/tokenclassification_conll2003_transformers"
            f" --max_eval_samples 100",
            shell=True,
        )

        eval_filename = f"{dir_path}/tokenclassification_conll2003_transformers/eval_results.json"

        with open(eval_filename, "r") as f:
            eval_dict = json.load(f)

        transformers_accuracy = eval_dict["eval_accuracy"]

        raw_datasets = load_dataset("conll2003")
        eval_dataset = raw_datasets["validation"].select([i for i in range(100)])

        model = AutoModelForTokenClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        pipe = pipeline(task="token-classification", model=model, tokenizer=tokenizer)

        e = evaluator(task="token-classification")
        results = e.compute(
            model_or_pipeline=pipe,
            data=eval_dataset,
            metric="seqeval",
            input_column="tokens",
            ref_column="ner_tags",
            strategy="simple",
        )

        evaluator_accuracy = results["overall_accuracy"]

        print("Evaluator accuracy:")
        print(evaluator_accuracy)

        print("Transformers example accuracy:")
        print(transformers_accuracy)

        self.assertEqual(transformers_accuracy, evaluator_accuracy)

import json
import os
import subprocess
import unittest
from datetime import datetime

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from evaluate import evaluator


class TestTextClassificationEvaluator(unittest.TestCase):
    def test_match_transformers_example(self):
        model_name = "howey/bert-base-uncased-sst2"

        tps = datetime.now().isoformat()
        dir_path = f"/tmp/{tps}_textclassification"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        subprocess.run(
            "git clone --depth 3 --filter=blob:none --sparse https://github.com/huggingface/transformers", shell=True
        )

        os.chdir("transformers")
        subprocess.run("git sparse-checkout set examples/pytorch/text-classification", shell=True)

        subprocess.run(
            f"python3 examples/pytorch/text-classification/run_glue.py"
            f" --model_name_or_path {model_name}"
            f" --task_name sst2"
            f" --do_eval"
            f" --max_seq_length 9999999999"  # rely on tokenizer.model_max_length for max_length
            f" --output_dir {dir_path}/textclassification_sst2_transformers"
            f" --max_eval_samples 200",
            shell=True,
        )

        eval_filename = f"{dir_path}/textclassification_sst2_transformers/eval_results.json"

        with open(eval_filename, "r") as f:
            eval_dict = json.load(f)

        transformers_accuracy = eval_dict["eval_accuracy"]

        raw_datasets = load_dataset("glue", "sst2")
        eval_dataset = raw_datasets["validation"].select([i for i in range(200)])

        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipe = pipeline(task="text-classification", model=model, tokenizer=tokenizer)

        e = evaluator(task="text-classification")
        results = e.compute(
            model_or_pipeline=pipe,
            data=eval_dataset,
            metric="accuracy",
            input_column="sentence",
            label_column="label",
            label_mapping={"LABEL_0": 0, "LABEL_1": 1},
            strategy="simple",
        )

        evaluator_accuracy = results["accuracy"]

        print("Evaluator accuracy:")
        print(evaluator_accuracy)

        print("Transformers example accuracy:")
        print(transformers_accuracy)

        self.assertEqual(transformers_accuracy, evaluator_accuracy)

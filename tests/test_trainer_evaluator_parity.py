import json
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, Trainer, TrainingArguments, pipeline

from evaluate import evaluator, load


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
        shutil.rmtree(self.dir_path, ignore_errors=True)

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

        pipe = pipeline(task="text-classification", model=model_name, tokenizer=model_name)

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
        # we can not compare to the Pytorch transformers example, that uses custom preprocessing on the images
        model_name = "douwekiela/resnet-18-finetuned-dogfood"
        dataset_name = "beans"
        max_eval_samples = 120

        raw_dataset = load_dataset(dataset_name, split="validation")
        eval_dataset = raw_dataset.select(range(max_eval_samples))

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)

        def collate_fn(examples):
            pixel_values = torch.stack(
                [torch.tensor(feature_extractor(example["image"])["pixel_values"][0]) for example in examples]
            )
            labels = torch.tensor([example["labels"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        metric = load("accuracy")
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=os.path.join(self.dir_path, "imageclassification_beans_transformers"),
                remove_unused_columns=False,
            ),
            train_dataset=None,
            eval_dataset=eval_dataset,
            compute_metrics=lambda p: metric.compute(
                predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
            ),
            tokenizer=None,
            data_collator=collate_fn,
        )

        metrics = trainer.evaluate()
        trainer.save_metrics("eval", metrics)

        with open(
            f"{os.path.join(self.dir_path, 'imageclassification_beans_transformers', 'eval_results.json')}", "r"
        ) as f:
            transformers_results = json.load(f)

        pipe = pipeline(task="image-classification", model=model_name, feature_extractor=model_name)

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

    def test_question_answering_parity(self):
        model_name = "mrm8488/bert-tiny-finetuned-squadv2"

        subprocess.run(
            "git sparse-checkout set examples/pytorch/question-answering",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        subprocess.run(
            f"python examples/pytorch/question-answering/run_qa.py"
            f" --model_name_or_path {model_name}"
            f" --dataset_name squad"
            f" --do_eval"
            f" --output_dir {os.path.join(self.dir_path, 'questionanswering_squad_transformers')}"
            f" --max_eval_samples 100"
            f" --max_seq_length 384",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        with open(
            f"{os.path.join(self.dir_path, 'questionanswering_squad_transformers', 'eval_results.json')}", "r"
        ) as f:
            transformers_results = json.load(f)

        eval_dataset = load_dataset("squad", split="validation[:100]")

        pipe = pipeline(task="question-answering", model=model_name, tokenizer=model_name, max_answer_len=30)

        e = evaluator(task="question-answering")
        evaluator_results = e.compute(
            model_or_pipeline=pipe,
            data=eval_dataset,
            metric="squad",
            strategy="simple",
        )

        self.assertEqual(transformers_results["eval_f1"], evaluator_results["f1"])
        self.assertEqual(transformers_results["eval_exact_match"], evaluator_results["exact_match"])

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

from .utils import slow


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

        eval_dataset = load_dataset("nyu-mll/glue", "sst2", split="validation[:80]")

        pipe = pipeline(task="text-classification", model=model_name, tokenizer=model_name)

        task_evaluator = evaluator(task="text-classification")
        evaluator_results = task_evaluator.compute(
            model_or_pipeline=pipe,
            data=eval_dataset,
            metric="accuracy",
            input_column="sentence",
            label_column="label",
            label_mapping={"negative": 0, "positive": 1},
            strategy="simple",
        )

        self.assertEqual(transformers_results["eval_accuracy"], evaluator_results["accuracy"])

    @slow
    def test_text_classification_parity_two_columns(self):
        model_name = "prajjwal1/bert-tiny-mnli"
        max_eval_samples = 150

        subprocess.run(
            "git sparse-checkout set examples/pytorch/text-classification",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        subprocess.run(
            f"python examples/pytorch/text-classification/run_glue.py"
            f" --model_name_or_path {model_name}"
            f" --task_name mnli"
            f" --do_eval"
            f" --max_seq_length 256"
            f" --output_dir {os.path.join(self.dir_path, 'textclassification_mnli_transformers')}"
            f" --max_eval_samples {max_eval_samples}",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        with open(
            f"{os.path.join(self.dir_path, 'textclassification_mnli_transformers', 'eval_results.json')}", "r"
        ) as f:
            transformers_results = json.load(f)

        eval_dataset = load_dataset("nyu-mll/glue", "mnli", split=f"validation_matched[:{max_eval_samples}]")

        pipe = pipeline(task="text-classification", model=model_name, tokenizer=model_name, max_length=256)

        task_evaluator = evaluator(task="text-classification")
        evaluator_results = task_evaluator.compute(
            model_or_pipeline=pipe,
            data=eval_dataset,
            metric="accuracy",
            input_column="premise",
            second_input_column="hypothesis",
            label_column="label",
            label_mapping={"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},
        )

        self.assertEqual(transformers_results["eval_accuracy"], evaluator_results["accuracy"])

    def test_image_classification_parity(self):
        # we can not compare to the Pytorch transformers example, that uses custom preprocessing on the images
        model_name = "douwekiela/resnet-18-finetuned-dogfood"
        dataset_name = "AI-Lab-Makerere/beans"
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

        task_evaluator = evaluator(task="image-classification")
        evaluator_results = task_evaluator.compute(
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
        model_name_v1 = "anas-awadalla/bert-tiny-finetuned-squad"
        model_name_v2 = "mrm8488/bert-tiny-finetuned-squadv2"

        subprocess.run(
            "git sparse-checkout set examples/pytorch/question-answering",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        # test squad_v1-like dataset
        subprocess.run(
            f"python examples/pytorch/question-answering/run_qa.py"
            f" --model_name_or_path {model_name_v1}"
            f" --dataset_name rajpurkar/squad"
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

        eval_dataset = load_dataset("rajpurkar/squad", split="validation[:100]")

        pipe = pipeline(
            task="question-answering",
            model=model_name_v1,
            tokenizer=model_name_v1,
            max_answer_len=30,
            padding="max_length",
        )

        task_evaluator = evaluator(task="question-answering")
        evaluator_results = task_evaluator.compute(
            model_or_pipeline=pipe,
            data=eval_dataset,
            metric="squad",
            strategy="simple",
        )

        self.assertEqual(transformers_results["eval_f1"], evaluator_results["f1"])
        self.assertEqual(transformers_results["eval_exact_match"], evaluator_results["exact_match"])

        # test squad_v2-like dataset
        subprocess.run(
            f"python examples/pytorch/question-answering/run_qa.py"
            f" --model_name_or_path {model_name_v2}"
            f" --dataset_name rajpurkar/squad_v2"
            f" --version_2_with_negative"
            f" --do_eval"
            f" --output_dir {os.path.join(self.dir_path, 'questionanswering_squadv2_transformers')}"
            f" --max_eval_samples 100"
            f" --max_seq_length 384",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        with open(
            f"{os.path.join(self.dir_path, 'questionanswering_squadv2_transformers', 'eval_results.json')}", "r"
        ) as f:
            transformers_results = json.load(f)

        eval_dataset = load_dataset("rajpurkar/squad_v2", split="validation[:100]")

        pipe = pipeline(
            task="question-answering",
            model=model_name_v2,
            tokenizer=model_name_v2,
            max_answer_len=30,
        )

        task_evaluator = evaluator(task="question-answering")
        evaluator_results = task_evaluator.compute(
            model_or_pipeline=pipe,
            data=eval_dataset,
            metric="squad_v2",
            strategy="simple",
            squad_v2_format=True,
        )

        self.assertEqual(transformers_results["eval_f1"], evaluator_results["f1"])
        self.assertEqual(transformers_results["eval_HasAns_f1"], evaluator_results["HasAns_f1"])
        self.assertEqual(transformers_results["eval_NoAns_f1"], evaluator_results["NoAns_f1"])

    def test_token_classification_parity(self):
        model_name = "hf-internal-testing/tiny-bert-for-token-classification"
        n_samples = 500

        subprocess.run(
            "git sparse-checkout set examples/pytorch/token-classification",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        subprocess.run(
            f"python examples/pytorch/token-classification/run_ner.py"
            f" --model_name_or_path {model_name}"
            f" --dataset_name areias/conll2003-generative"
            f" --do_eval"
            f" --output_dir {os.path.join(self.dir_path, 'tokenclassification_conll2003_transformers')}"
            f" --max_eval_samples {n_samples}",
            shell=True,
            cwd=os.path.join(self.dir_path, "transformers"),
        )

        with open(
            os.path.join(self.dir_path, "tokenclassification_conll2003_transformers", "eval_results.json"), "r"
        ) as f:
            transformers_results = json.load(f)

        eval_dataset = load_dataset("areias/conll2003-generative", split=f"validation[:{n_samples}]")

        pipe = pipeline(task="token-classification", model=model_name)

        e = evaluator(task="token-classification")
        evaluator_results = e.compute(
            model_or_pipeline=pipe,
            data=eval_dataset,
            metric="seqeval",
            input_column="tokens",
            label_column="ner_tags",
            strategy="simple",
        )

        self.assertEqual(transformers_results["eval_accuracy"], evaluator_results["overall_accuracy"])
        self.assertEqual(transformers_results["eval_f1"], evaluator_results["overall_f1"])

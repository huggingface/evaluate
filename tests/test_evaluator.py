# Copyright 2022 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

from unittest import TestCase

from datasets import load_dataset, load_metric
from transformers import pipeline

from evaluate import TextClassificationEvaluator, evaluator


class TestEvaluator(TestCase):
    def test_evaluator(self):
        pipe = pipeline("text-classification")
        accuracy = load_metric("accuracy")
        ds = load_dataset("imdb")
        ds = ds["test"].shuffle().select(range(32))  # just for speed
        ds = ds.rename_columns({"text": "inputs", "label": "references"})

        ev = evaluator("text-classification")
        print(ev.compute(pipe, ds, label_mapping={"NEGATIVE": 0, "POSITIVE": 1}))
        print(
            ev.compute(
                model_or_pipeline="huggingface/prunebert-base-uncased-6-finepruned-w-distil-mnli",
                data=ds,
                label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},
                strategy="bootstrap",
                n_resamples=99,
            )
        )

        print(
            ev.compute(
                data=ds,
                metric="accuracy",
                label_mapping={"NEGATIVE": 0, "POSITIVE": 1},
            )
        )

        ev3 = TextClassificationEvaluator()
        print(
            ev3.compute(
                model_or_pipeline=pipe,
                data=ds,
                metric=accuracy,
                label_mapping={"NEGATIVE": 0, "POSITIVE": 1},
            )
        )

        ev_err = evaluator("question_answering")
        self.assertEqual(1, 2)

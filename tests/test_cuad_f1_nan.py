# Copyright 2026 The HuggingFace Evaluate Authors.
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
import importlib.util
import math
from pathlib import Path


CUAD_SCORE_PATH = Path(__file__).resolve().parents[1] / "metrics" / "cuad" / "compute_score.py"


def _load_compute_score():
    spec = importlib.util.spec_from_file_location("cuad_compute_score", CUAD_SCORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


compute_score = _load_compute_score()


def _dataset(answers):
    """One paragraph with one question per entry of `answers`."""
    return [
        {
            "paragraphs": [
                {
                    "qas": [
                        {"id": f"qa{i}", "answers": [{"text": text, "answer_start": 0} for text in texts]}
                        for i, texts in enumerate(answers, start=1)
                    ]
                }
            ]
        }
    ]


def test_empty_prediction_list_does_not_poison_corpus_f1():
    # qa1 is answered with an empty prediction_text list, qa2 is answered correctly.
    scores = compute_score(_dataset([["alpha"], ["beta"]]), {"qa1": [], "qa2": ["beta"]})

    assert math.isfinite(scores["f1"]), f"corpus f1 is not a number: {scores['f1']}"
    assert scores["f1"] == 50.0
    assert scores["exact_match"] == 50.0


def test_empty_prediction_list_scores_like_an_omitted_prediction():
    # Same information as above, but qa1 is left out of the predictions dict.
    scores = compute_score(_dataset([["alpha"], ["beta"]]), {"qa2": ["beta"]})

    assert scores["f1"] == 50.0
    assert scores["exact_match"] == 50.0


def test_empty_gold_and_empty_prediction_does_not_poison_corpus_f1():
    # A question with no gold answers and no prediction is a 0/0 sample.
    scores = compute_score(_dataset([[]]), {"qa1": []})

    assert math.isfinite(scores["f1"]), f"corpus f1 is not a number: {scores['f1']}"
    assert scores["f1"] == 0.0

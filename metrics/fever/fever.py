# Copyright 2021 The HuggingFace Evaluate Authors.
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

"""FEVER (Fact Extraction and VERification) metric."""

import datasets

import evaluate


_CITATION = """\
@inproceedings{thorne2018fever,
  title={FEVER: Fact Extraction and VERification},
  author={Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={809--819},
  year={2018}
}
"""
_DESCRIPTION = """\
The FEVER (Fact Extraction and VERification) metric evaluates the performance of systems that verify factual claims against evidence retrieved from Wikipedia.

It consists of three main components:
- **Label accuracy**: measures how often the predicted claim label (SUPPORTED, REFUTED, or NOT ENOUGH INFO) matches the gold label.
- **FEVER score**: considers a prediction correct only if the label is correct *and* at least one complete gold evidence set is retrieved.
- **Evidence F1**: computes the micro-averaged precision, recall, and F1 between predicted and gold evidence sentences.

The FEVER score is the official leaderboard metric used in the FEVER shared tasks.
"""
_KWARGS_DESCRIPTION = """
Computes the FEVER evaluation metrics.

Args:
    predictions (list of dict): Each prediction should be a dictionary with:
        - "label" (str): the predicted claim label.
        - "evidence" (list of str): the predicted evidence sentences.
    references (list of dict): Each reference should be a dictionary with:
        - "label" (str): the gold claim label.
        - "evidence_sets" (list of list of str): all possible gold evidence sets.

Returns:
    A dictionary containing:
        - 'label_accuracy': proportion of claims with correctly predicted labels.
        - 'fever_score': proportion of claims where both the label and at least one full gold evidence set are correct.
        - 'evidence_precision': micro-averaged precision of evidence retrieval.
        - 'evidence_recall': micro-averaged recall of evidence retrieval.
        - 'evidence_f1': micro-averaged F1 of evidence retrieval.

Example:
    >>> predictions = [{"label": "SUPPORTED", "evidence": ["E1", "E2"]}]
    >>> references = [{"label": "SUPPORTED", "evidence_sets": [["E1", "E2"], ["E3", "E4"]]}]
    >>> fever = evaluate.load("fever")
    >>> results = fever.compute(predictions=predictions, references=references)
    >>> print(results)
    {'label_accuracy': 1.0, 'fever_score': 1.0, 'evidence_precision': 1.0, 'evidence_recall': 1.0, 'evidence_f1': 1.0}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class FEVER(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {
                        "label": datasets.Value("string"),
                        "evidence": datasets.Sequence(datasets.Value("string")),
                    },
                    "references": {
                        "label": datasets.Value("string"),
                        "evidence_sets": datasets.Sequence(
                            datasets.Sequence(datasets.Value("string"))
                        ),
                    },
                }
            ),
            reference_urls=[
                "https://fever.ai/dataset/",
                "https://arxiv.org/abs/1803.05355",
            ],
        )

    def _compute(self, predictions, references):
        """
        Computes FEVER metrics:
        - Label accuracy
        - FEVER score (label + complete evidence set)
        - Evidence precision, recall, and F1 (micro-averaged)
        """
        total = len(predictions)
        label_correct, fever_correct = 0, 0
        total_overlap, total_pred, total_gold = 0, 0, 0

        for pred, ref in zip(predictions, references):
            pred_label = pred["label"]
            pred_evidence = set(e.strip().lower() for e in pred["evidence"])
            gold_label = ref["label"]
            gold_sets = []
            for s in ref["evidence_sets"]:
                gold_sets.append([e.strip().lower() for e in s])

            if pred_label == gold_label:
                label_correct += 1
                for g_set in gold_sets:
                    if set(g_set).issubset(pred_evidence):
                        fever_correct += 1
                        break

            gold_evidence = set().union(*gold_sets) if gold_sets else set()
            overlap = len(gold_evidence.intersection(pred_evidence))
            total_overlap += overlap
            total_pred += len(pred_evidence)
            total_gold += len(gold_evidence)

        precision = (total_overlap / total_pred) if total_pred else 0
        recall = (total_overlap / total_gold) if total_gold else 0
        evidence_f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        fever_score = fever_correct / total if total else 0
        label_accuracy = label_correct / total if total else 0

        return {
            "label_accuracy": label_accuracy,
            "fever_score": fever_score,
            "evidence_precision": precision,
            "evidence_recall": recall,
            "evidence_f1": evidence_f1,
        }

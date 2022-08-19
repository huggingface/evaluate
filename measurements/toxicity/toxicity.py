# Copyright 2020 The HuggingFace Evaluate Authors.
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

""" Toxicity detection measurement. """

import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import evaluate


logger = evaluate.logging.get_logger(__name__)


_CITATION = """
@inproceedings{vidgen2021lftw,
  title={Learning from the Worst: Dynamically Generated Datasets to Improve Online Hate Detection},
  author={Bertie Vidgen and Tristan Thrush and Zeerak Waseem and Douwe Kiela},
  booktitle={ACL},
  year={2021}
}
"""

_DESCRIPTION = """\
The toxicity measurement aims to quantify the toxicity of the input texts using a hate speech classification model trained for the task
"""

_KWARGS_DESCRIPTION = """
Compute the toxicity of the input sentences.

Args:
    `predictions` (list of str): prediction/candidate sentences
    `toxic_label` (optional): the toxic label that you want to detect, depending on the labels that the model has been trained on.
        This can be found using the `model.config.id2label` function.
    `aggregation` (optional): determines the type of aggregation performed on the data. If set to `None`, the scores for each prediction are returned.
     Otherwise:
        - 'maximum': returns the maximum toxicity over all predictions
        - 'ratio': the percentage of predictions with toxicity >= 0.5.
Returns:
    `toxicity`: a list of toxicity scores, one for each sentence in `predictions` (default behavior)
    `max_toxicity`: the maximum toxicity over all scores (if `aggregation` = `maximum`)
    `toxicity_ratio`": the percentage of predictions with toxicity >= 0.5 (if `aggregation` = `ratio`)

Examples:
    Example 1:
        >>> toxicity = evaluate.load("toxicity", module_type="measurement")
        >>> input_texts = ["she is very mean", "he is a douchebag", "you're ugly"]
        >>> results = toxicity.compute(predictions=input_texts)
        >>> print(results)
        {'toxicity': [0.00013419731112662703, 0.856372594833374, 0.0020856475457549095]}

    Example 2:
        >>> toxicity = evaluate.load("toxicity", module_type="measurement")
        >>> input_texts = ["she is very mean", "he is a douchebag", "you're ugly"]
        >>> results = toxicity.compute(predictions=input_texts, aggregation = "ratio")
        >>> print(results)
        {'toxicity_ratio': 0.3333333333333333}

    Example 3:
        >>> toxicity = evaluate.load("toxicity", module_type="measurement")
        >>> input_texts = ["she is very mean", "he is a douchebag", "you're ugly"]
        >>> results = toxicity.compute(predictions=input_texts, aggregation = "maximum")
        >>> print(results)
        {'max_toxicity': 0.856372594833374}

    Example 4:
        >>> toxicity = evaluate.load("toxicity", module_type="measurement", 'DaNLP/da-electra-hatespeech-detection')
        >>> input_texts = ["she is very mean", "he is a douchebag", "you're ugly"]
        >>> results = toxicity.compute(predictions=input_texts, toxic_label='offensive')
        >>> print(results)
        {'toxicity': [0.004464445170015097, 0.020320769399404526, 0.01239820383489132]}
"""


def toxicity(preds, toxic_classifier, toxic_label):
    toxic_scores = []
    for pred in preds:
        pred_toxic = toxic_classifier(str(pred))
        hate_toxic = [r["score"] for r in pred_toxic if r["label"] == toxic_label][0]
        toxic_scores.append(hate_toxic)
    return toxic_scores


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Toxicity(evaluate.Measurement):
    def _info(self):
        return evaluate.MeasurementInfo(
            module_type="measurement",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )

    def _download_and_prepare(self, dl_manager):
        if self.config_name == "default":
            logger.warning("Using default facebook/roberta-hate-speech-dynabench-r4-target checkpoint")
            self.toxic_classifier = pipeline(
                "text-classification",
                model="facebook/roberta-hate-speech-dynabench-r4-target",
                top_k=2,
                truncation=True,
            )
        else:
            self.toxic_classifier = pipeline("text-classification", model=self.config_name, top_k=2, truncation=True)

    def _compute(
        self,
        predictions,
        aggregation="all",
        toxic_label="hate",
    ):
        scores = toxicity(predictions, self.toxic_classifier, toxic_label)
        if aggregation == "ratio":
            return {"toxicity_ratio": sum(i >= 0.5 for i in scores) / len(scores)}
        elif aggregation == "maximum":
            return {"max_toxicity": max(scores)}
        else:
            return {"toxicity": scores}

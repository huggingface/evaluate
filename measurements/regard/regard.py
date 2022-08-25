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

""" Regard measurement. """

from statistics import mean
from collections import defaultdict
from operator import itemgetter
import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import evaluate


logger = evaluate.logging.get_logger(__name__)


_CITATION = """
@article{https://doi.org/10.48550/arxiv.1909.01326,
  doi = {10.48550/ARXIV.1909.01326},
  url = {https://arxiv.org/abs/1909.01326},
  author = {Sheng, Emily and Chang, Kai-Wei and Natarajan, Premkumar and Peng, Nanyun},
  title = {The Woman Worked as a Babysitter: On Biases in Language Generation},
  publisher = {arXiv},
  year = {2019}
}

"""

_DESCRIPTION = """\
Regard aims to measure language polarity towards and social perceptions of a demographic (e.g. gender, race, sexual orientation).
"""

_KWARGS_DESCRIPTION = """
Compute the regard of the input sentences.

Args:
    `predictions` (list of str): prediction/candidate sentences, e.g. sentences describing a given demographic group.
    `references` (list of str): reference/comparison sentences, e.g. sentences decribing a different demographic group to compare against.
    `aggregation` (str) (optional): determines the type of aggregation performed on the data.
    If set to `None`, the difference between the regard scores for the two categories is returned.
     Otherwise:
        - 'average' : returns the average regard for each category (negative, positive, neutral, other) for each group
        - 'maximum': returns the maximum regard for each group

Returns:
    `regard_difference`: the difference between the regard scores for the two groups
    `average_predictions_regard` and 'average_references_regard': the average regard for each category (negative, positive, neutral, other)  (if `aggregation` = `average`)
    `max_predictions_regard` and 'max_references_regard': the maximum regard for each group  (if `aggregation` = `maximum`)

Examples:

# TODO:

"""


def regard(preds, refs, regard_classifier):
    pred_scores= defaultdict(list)
    ref_scores = defaultdict(list)
    pred_regard = regard_classifier(preds)
    ref_regard =  regard_classifier(refs)
    for pred in pred_regard:
        for pred_score in pred:
            pred_scores[pred_score['label']].append(pred_score['score'])
    for ref in ref_regard:
        for ref_score in ref:
            ref_scores[ref_score['label']].append(ref_score['score'])
    return dict(pred_scores), dict(ref_scores)


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Regard(evaluate.Measurement):
    def _info(self):
        return evaluate.MeasurementInfo(
            module_type="measurement",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[],
            reference_urls=[],
        )

    def _download_and_prepare(self, dl_manager):
        regard_tokenizer = AutoTokenizer.from_pretrained("sasha/regardv3")
        regard_model = AutoModelForSequenceClassification.from_pretrained("sasha/regardv3")
        self.regard_classifier = pipeline(
            "text-classification", model=regard_model, top_k=4, tokenizer=regard_tokenizer, truncation=True
        )

    def _compute(
        self,
        predictions,
        references,
        aggregation=None,
    ):
        pred_regard, ref_regard = regard(predictions, references, self.regard_classifier)
        pred_average, ref_average = {}, {}
        for k, v in pred_regard.items():
            pred_average[k] = mean(v)
        for k, v in ref_regard.items():
            ref_average[k] = mean(v)
        if aggregation == "maximum":
            pred_max, ref_max = {}, {}
            for k, v in pred_regard.items():
                pred_max[k] = max(v)
            for k, v in ref_regard.items():
                ref_max[k] = max(v)
            return {"max_predictions_regard": max(pred_max.items(), key=itemgetter(1)),
                    "max_references_regard":  max(ref_max.items(), key=itemgetter(1))}
        elif aggregation == "average":
            return {"average_predictions_regard": pred_average, "average_references_regard": ref_average}
        else:
            return {"regard_difference": {key: pred_average[key] - ref_average.get(key, 0) for key in pred_average}}

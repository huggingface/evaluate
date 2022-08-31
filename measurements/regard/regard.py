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

from collections import defaultdict
from operator import itemgetter
from statistics import mean

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
    `data` (list of str): prediction/candidate sentences, e.g. sentences describing a given demographic group.
    `references` (list of str): reference/comparison sentences, e.g. sentences describing a different demographic group to compare against.
    `aggregation` (str) (optional): determines the type of aggregation performed.
    If set to `None`, the difference between the regard scores for the two categories is returned.
     Otherwise:
        - 'average' : returns the average regard for each category (negative, positive, neutral, other) for each group
        - 'maximum': returns the maximum regard for each group

Returns:
    `regard_difference`: the difference between the regard scores for the two groups
    `average_data_regard` and 'average_references_regard': the average regard for each category (negative, positive, neutral, other)  (if `aggregation` = `average`)
    `max_data_regard` and 'max_references_regard': the maximum regard for each group  (if `aggregation` = `maximum`)

Examples:

Example 1 (default behavior):
>>> regard = evaluate.load("regard", module_type="measurement")
>>> group1 = ['xyz are described as mean', 'xyz are thought of as being too ambitious']
>>> group2 = ['xyz are known for making too much noise', 'xyz are described as often violent']
>>> results = regard.compute(data = group1, references = group2)
>>> print({k: round(v, 2) for k, v in results['regard_difference'].items()})
{'neutral': 0.46, 'positive': 0.01, 'negative': -0.46, 'other': -0.01}

Example 2 (returns the maximum toxicity score):
>>> regard = evaluate.load("regard", module_type="measurement")
>>> group1 = ['the girls were mean', 'women are too ambitious']
>>> group2 = ['boys make too much noise', 'men are often violent']
>>> results = regard.compute(data = group1, references = group2, aggregation = "maximum")
>>> print({k: round(v,3) for k,v in results.items()})
{'max_predictions_regard': 0.972, 'max_references_regard': 0.979}

Example 3 (returns the average toxicity score):
>>> regard = evaluate.load("regard", module_type="measurement")
>>> group1 = ['the girls were mean', 'women are too ambitious']
>>> group2 = ['boys make too much noise', 'men are often violent']
>>> results = regard.compute(data = group1, references = group2, aggregation = "average")
>>> print({k: round(v, 2) for k, v in results['average_data_regard'].items()})
{'neutral': 0.48, 'positive': 0.01, 'negative': 0.5, 'other': 0.01}
>>> print({k: round(v, 2) for k, v in results['average_references_regard'].items()})
{'negative': 0.96, 'other': 0.02, 'neutral': 0.02, 'positive': 0.0}
"""


def regard(preds, refs, regard_classifier):
    pred_scores = defaultdict(list)
    ref_scores = defaultdict(list)
    pred_regard = regard_classifier(preds)
    ref_regard = regard_classifier(refs)
    for pred in pred_regard:
        for pred_score in pred:
            pred_scores[pred_score["label"]].append(pred_score["score"])
    for ref in ref_regard:
        for ref_score in ref:
            ref_scores[ref_score["label"]].append(ref_score["score"])
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
                    "data": datasets.Value("string", id="sequence"),
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
        data,
        references,
        aggregation=None,
    ):
        pred_regard, ref_regard = regard(data, references, self.regard_classifier)
        pred_mean = {k: mean(v) for k, v in pred_regard.items()}
        ref_mean = {k: mean(v) for k, v in ref_regard.items()}
        if aggregation == "maximum":
            return {
                "max_predictions_regard": max([max(v) for v in pred_regard.values()]),
                "max_references_regard": max([max(v) for v in ref_regard.values()]),
            }
        elif aggregation == "average":
            return {"average_data_regard": pred_mean, "average_references_regard": ref_mean}
        else:
            return {"regard_difference": {key: pred_mean[key] - ref_mean.get(key, 0) for key in pred_mean}}

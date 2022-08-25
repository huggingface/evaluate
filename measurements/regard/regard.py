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
    `predictions` (list of str): prediction/candidate sentences
    `aggregation` (str) (optional): determines the type of aggregation performed on the data.
    If set to `None`, the scores for each sentences are returned.
     Otherwise:
        - 'average' : returns the average regard for each category (negative, positive, neutral, other)
        - 'maximum': returns the maximum regard for each category

Returns:
    `regard`: a list of regard scores, one for each sentence in `predictions` (default behavior)
    `average_regard`: the average regard for each category (negative, positive, neutral, other)  (if `aggregation` = `average`)
    `max_regard`: the maximum regard for each category  (if `aggregation` = `maximum`)

Examples:
    Example 1 (default behavior):
    >>> regard = evaluate.load("regard", module_type="measurement")
    >>> input_texts = ["these girls are silly", "these boys are lost"]
    >>> results = regard.compute(predictions=input_texts)
    >>> print(results)
    {'regard': [[{'label': 'negative', 'score': 0.6691194772720337}, {'label': 'other', 'score': 0.22687028348445892}, {'label': 'neutral', 'score': 0.0852026417851448}, {'label': 'positive', 'score': 0.018807603046298027}], [{'label': 'neutral', 'score': 0.942646861076355}, {'label': 'positive', 'score': 0.02632979303598404}, {'label': 'negative', 'score': 0.020616641268134117}, {'label': 'other', 'score': 0.010406642220914364}]]}
Example 2 (returns the maximum toxicity score):
    >>> regard = evaluate.load("regard", module_type="measurement")
    >>> input_texts = ["these girls are silly", "these boys are lost"]
    >>> results = toxicity.compute(predictions=input_texts, aggregation = "maximum")
    >>> print(results)
    {'max_regard': {'negative': 0.6691194772720337, 'positive': 0.02632979303598404, 'neutral': 0.942646861076355, 'other': 0.22687028348445892}}
Example 3 (returns the average toxicity score):
    >>> regard = evaluate.load("regard", module_type="measurement")
    >>> input_texts = ["these girls are silly", "these boys are lost"]
    >>> results = toxicity.compute(predictions=input_texts, aggregation = "average")
    >>> print(results)
    {'average_regard': {'negative': 0.3448680592700839, 'positive': 0.022568698041141033, 'neutral': 0.5139247514307499, 'other': 0.11863846285268664}}
"""


def regard(preds, regard_classifier):
    neg_scores = []
    pos_scores = []
    neutral_scores = []
    other_scores = []
    regard_scores = {}
    all_regard = []
    for pred in preds:
        regard = regard_classifier(pred)
        all_regard.append(regard)
        for s in regard:
            if s["label"] == "negative":
                neg_scores.append(s["score"])
            elif s["label"] == "positive":
                pos_scores.append(s["score"])
            elif s["label"] == "neutral":
                neutral_scores.append(s["score"])
            else:
                other_scores.append(s["score"])
        regard_scores["negative"] = neg_scores
        regard_scores["positive"] = pos_scores
        regard_scores["neutral"] = neutral_scores
        regard_scores["other"] = other_scores
    return all_regard, regard_scores


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
        aggregation="all",
    ):
        all, scores = regard(predictions, self.regard_classifier)
        if aggregation == "maximum":
            max_dict = {}
            for i in scores.items():
                max_dict[i[0]] = max(i[1])
            return {"max_regard": max_dict}
        elif aggregation == "average":
            av_dict = {}
            for i in scores.items():
                av_dict[i[0]] = mean(i[1])
            return {"average_regard": av_dict}
        else:
            return {"regard": all}

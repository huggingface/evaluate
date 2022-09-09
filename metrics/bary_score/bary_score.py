# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Bary Score Metric"""

import datasets
from metrics.bary_score.score import BaryScoreMetric
import evaluate


_CITATION = """\
@inproceedings{colombo-etal-2021-automatic,
    title = "Automatic Text Evaluation through the Lens of {W}asserstein Barycenters",
    author = "Colombo, Pierre  and Staerman, Guillaume  and Clavel, Chlo{\'e}  and Piantanida, Pablo",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    year = "2021",
    pages = "10450--10466"
}
"""

_DESCRIPTION = """\
BaryScore is a multi-layers metric based on pretrained contextualized representations. Similar to MoverScore, it aggregates the layers of Bert before computing a similarity score. By modelling the layer output of deep contextualized embeddings as a probability distribution rather than by a vector embedding; BaryScore aggregates the different outputs through the Wasserstein space topology. MoverScore (right) leverages the information available in other layers by aggregating the layers using a power mean and then use a Wasserstein distance ().
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: list of string sentences has to be.
    references: list of string sentences has to be.
Examples:
    >>> barry_score = evaluate.load("bary_score")
    >>> references = ['I like my cakes very much','I hate these cakes!']
    >>> predictions = ['I like my cakes very much','I like my cakes very much']
    >>> results = barry_score.compute(predictions=predictions, references=references)
    >>> print(results)
    {'baryscore_W': [2.220446049250313e-16, 0.4936737487362536], 'baryscore_SD_10': [0.9234963490510808, 1.0454139159538949], 'baryscore_SD_1': [0.7360736368883636, 0.9437504927697342], 'baryscore_SD_5': [0.9074753479358628, 1.036207173522238], 'baryscore_SD_0.1': [0.0007089180091671455, 0.5100520249124377], 'baryscore_SD_0.5': [0.4623988987972563, 0.8098911748431552], 'baryscore_SD_0.01': [2.220446049250317e-16, 0.49367374955620186], 'baryscore_SD_0.001': [2.220446049250313e-16, 3.118914392068233e-08]}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BaryScore(evaluate.EvaluationModule):
    def _info(self):
        return evaluate.EvaluationModuleInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/PierreColombo/nlg_eval_via_simi_measures"],
            reference_urls=["https://arxiv.org/pdf/2108.12463.pdf"],
        )

    def _compute(
        self, predictions, references, model_name="bert-base-uncased", last_layers=5, use_idfs=True, sinkhorn_ref=0.01
    ):
        metric_call = BaryScoreMetric(
            model_name=model_name, last_layers=last_layers, use_idfs=use_idfs, sinkhorn_ref=sinkhorn_ref
        )
        metric_call.prepare_idfs(references, predictions)
        result = metric_call.evaluate_batch(references, predictions)
        return result

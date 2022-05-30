# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Module to compute TREC evaluation scores."""

import datasets
import pandas as pd
from trectools import TrecEval, TrecQrel, TrecRun

import evaluate


_CITATION = """\
@inproceedings{palotti2019,
 author = {Palotti, Joao and Scells, Harrisen and Zuccon, Guido},
 title = {TrecTools: an open-source Python library for Information Retrieval practitioners involved in TREC-like campaigns},
 series = {SIGIR'19},
 year = {2019},
 location = {Paris, France},
 publisher = {ACM}
}
"""

# TODO: Add description of the module here
_DESCRIPTION = """\
The TREC Eval metric combines a number of information retrieval metrics such as \
precision and nDCG. It is used to score rankings of retrieved documents with reference values."""


# TODO: Add description of the arguments of the module here
_KWARGS_DESCRIPTION = """
Calculates TREC evaluation scores based on a run and qrel.
Args:
    predictions: list containing a single run.
    references: list containing a single qrel.
Returns:
    dict: TREC evaluation scores.
Examples:
    >>> trec = evaluate.load("trec_eval")
    >>> qrel = {
    ...     "query": [0],
    ...     "q0": ["0"],
    ...     "docid": ["doc_1"],
    ...     "rel": [2]
    ... }
    >>> run = {
    ...     "query": [0, 0],
    ...     "q0": ["q0", "q0"],
    ...     "docid": ["doc_2", "doc_1"],
    ...     "rank": [0, 1],
    ...     "score": [1.5, 1.2],
    ...     "system": ["test", "test"]
    ... }
    >>> results = trec.compute(references=[qrel], predictions=[run])
    >>> print(results["P@5"])
    0.2
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TRECEval(evaluate.EvaluationModule):
    """Compute TREC evaluation scores."""

    def _info(self):
        return evaluate.EvaluationModuleInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {
                        "query": datasets.Sequence(datasets.Value("int64")),
                        "q0": datasets.Sequence(datasets.Value("string")),
                        "docid": datasets.Sequence(datasets.Value("string")),
                        "rank": datasets.Sequence(datasets.Value("int64")),
                        "score": datasets.Sequence(datasets.Value("float")),
                        "system": datasets.Sequence(datasets.Value("string")),
                    },
                    "references": {
                        "query": datasets.Sequence(datasets.Value("int64")),
                        "q0": datasets.Sequence(datasets.Value("string")),
                        "docid": datasets.Sequence(datasets.Value("string")),
                        "rel": datasets.Sequence(datasets.Value("int64")),
                    },
                }
            ),
            homepage="https://github.com/joaopalotti/trectools",
        )

    def _compute(self, references, predictions):
        """Returns the TREC evaluation scores."""

        if len(predictions) > 1 or len(references) > 1:
            raise ValueError(
                f"You can only pass one prediction and reference per evaluation. You passed {len(predictions)} prediction(s) and {len(references)} reference(s)."
            )

        df_run = pd.DataFrame(predictions[0])
        df_qrel = pd.DataFrame(references[0])

        trec_run = TrecRun()
        trec_run.filename = "placeholder.file"
        trec_run.run_data = df_run

        trec_qrel = TrecQrel()
        trec_qrel.filename = "placeholder.file"
        trec_qrel.qrels_data = df_qrel

        trec_eval = TrecEval(trec_run, trec_qrel)

        result = {}
        result["runid"] = trec_eval.run.get_runid()
        result["num_ret"] = trec_eval.get_retrieved_documents(per_query=False)
        result["num_rel"] = trec_eval.get_relevant_documents(per_query=False)
        result["num_rel_ret"] = trec_eval.get_relevant_retrieved_documents(per_query=False)
        result["num_q"] = len(trec_eval.run.topics())
        result["map"] = trec_eval.get_map(depth=10000, per_query=False, trec_eval=True)
        result["gm_map"] = trec_eval.get_geometric_map(depth=10000, trec_eval=True)
        result["bpref"] = trec_eval.get_bpref(depth=1000, per_query=False, trec_eval=True)
        result["Rprec"] = trec_eval.get_rprec(depth=1000, per_query=False, trec_eval=True)
        result["recip_rank"] = trec_eval.get_reciprocal_rank(depth=1000, per_query=False, trec_eval=True)

        for v in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
            result[f"P@{v}"] = trec_eval.get_precision(depth=v, per_query=False, trec_eval=True)
        for v in [5, 10, 15, 20, 30, 100, 200, 500, 1000]:
            result[f"NDCG@{v}"] = trec_eval.get_ndcg(depth=v, per_query=False, trec_eval=True)

        return result

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
"""TODO: Add a description here."""

import evaluate
import datasets


# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:module,
title = {A great new module},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the module here
_DESCRIPTION = """\
This new module is designed calculate kendall's tau distance between predictions and references.
It is also known as bubble sort distance. 
It is equivalent to number of adjacent swaps required to convert predictions to references.
"""


# TODO: Add description of the arguments of the module here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, usoing kendall's tau distance.
Args:
    predictions: list of predictions to score. Each predictions
        should be a string or tokens or int. The predictions should be unique.
    references: list of reference for each prediction. Each reference
        should be a string or tokens or int. The values in predictions and references should be the same.
Returns:
    kendall_tau_distance: Kendell's tau distance between predictions and references
    normalized_kendall_tau_distance: Kendell's tau distance between predictions and references normalized by the number of pairs

Exceptions:
    AssertionError: If the predictions are not unique or if the values in predictions and references are not the same

Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> kendall_tau_distance = evaluate.load("kendall_tau_distance")
    >>> results = kendall_tau_distance.compute(references=[0, 1], predictions=[1, 0])
    >>> print(results)
    {'kendall_tau_distance': 1.0, 'normalized_kendall_tau_distance': 1.0}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class kendalltaudistance(evaluate.Metric):
    def _info(self):
        # TODO: Specifies the evaluate.EvaluationModuleInfo object
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int64"),
                    "references": datasets.Value("int64"),
                }
            )
        )

    def _compute(self, predictions, references):
        """Returns the scores"""
        # TODO: Compute the different scores of the module

        n = len(predictions)

        assert len(set(predictions)) == n, "The predictions should be unique"
        assert set(predictions) == set(
            references
        ), "The values in predictions and references should be the same"
        n_discordant_pairs = 0

        for i in range(len(predictions)):
            j = references.index(predictions[i])
            n_discordant_pairs += len(
                set(predictions[:i]).intersection(set(references[j:]))
            ) + len(set(predictions[i + 1 :]).intersection(set(references[:j])))

        n_discordant_pairs = n_discordant_pairs / 2

        num_pairs = n * (n - 1) / 2

        return {
            "kendall_tau_distance": n_discordant_pairs,
            "normalized_kendall_tau_distance": n_discordant_pairs / num_pairs,
        }

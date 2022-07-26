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
"""Label Distribution Measurement."""

import evaluate
import datasets
from collections import Counter
import pandas as pd


_DESCRIPTION = """
Returns the label ratios of the dataset labels, as well as a scalar for skewness.
"""

_KWARGS_DESCRIPTION = """
Args:
    `data`: a list containing the data labels

Returns:
    `label_distribution` (`dict`) :  a dictionary containing two sets of keys and values: `labels`, which includes the list of labels contained in the dataset, and `fractions`, which includes the fraction of each label.
    `label_skew` (`scalar`)  (optional) : the asymmetry of the label distribution.
Examples:
    >>> data = [1, 0, 1, 1, 0, 1, 0]
    >>> distribution = evaluate.load("label_distribution")
    >>> results = distribution.compute(data=data)
    >>> print(results)
    {'label_distribution': {'labels': [1, 0], 'fractions': [0.5714285714285714, 0.42857142857142855]}}

    >>> from datasets import load_dataset
    >>> imdb = load_dataset('imdb', split = 'test')
    >>> distribution = evaluate.load("label_distribution")
    >>> results = distribution.compute(data=imdb['label'], return_skew=True)
    >>> print(results)
    {'label_distribution': {'labels': [0, 1], 'fractions': [0.5, 0.5]}, 'label_skew': 0.0}
"""

# TODO: Add BibTeX citation
_CITATION = "TODO"

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class LabelDistribution(evaluate.Measurement):
    def _info(self):
        return evaluate.MeasurementInfo(
            module_type="measurement",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
            datasets.Features(
                {
                    "data": datasets.Value("int32")
                }
                ),
                datasets.Features(
                    {
                    "data": datasets.Value("string")
                    }
                ),
            ]
        )

    def _compute(self, data, return_skew = False):
        """Returns the fraction of each label present in the data"""
        c = Counter(data)
        #label_distribution = [(i, c[i] / len(data)) for i in c]
        label_distribution = {"labels" : [k for k in c.keys()], "fractions" : [f/len(data) for f in c.values()]}
        if return_skew == True:
            sr= pd.Series(data)
            skew = sr.skew()
            return {"label_distribution": label_distribution, "label_skew": skew}
        else:
             return {"label_distribution": label_distribution}

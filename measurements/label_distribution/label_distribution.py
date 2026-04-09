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

from collections import Counter

import datasets
import numpy as np
from scipy.stats import entropy

import evaluate


_DESCRIPTION = """
Returns the label ratios of the dataset labels, as well as the Shannon entropy of the label distribution.
"""

_KWARGS_DESCRIPTION = """
Args:
    `data`: a list containing the data labels

Returns:
    `label_distribution` (`dict`) :  a dictionary containing two sets of keys and values: `labels`, which includes the list of labels contained in the dataset, and `fractions`, which includes the fraction of each label.
    `label_entropy` (`float`) : the Shannon entropy of the label distribution (in nats). Maximized at log(k) for k classes when labels are uniformly distributed, and 0 when all labels are the same.
    `label_entropy_normalized` (`float`) : the Shannon entropy normalized by log(k), giving a value between 0 and 1. A value of 1.0 means perfectly balanced; a value close to 0 means highly imbalanced.
Examples:
    >>> data = [1, 0, 1, 1, 0, 1, 0]
    >>> distribution = evaluate.load("label_distribution")
    >>> results = distribution.compute(data=data)
    >>> print(results)
    {'label_distribution': {'labels': [1, 0], 'fractions': [0.5714285714285714, 0.42857142857142855]}, 'label_entropy': 0.6829081047004717, 'label_entropy_normalized': 0.9852281360342515}
"""

_CITATION = """\
@ARTICLE{2020SciPy-NMeth,
  author  = {Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and
            Haberland, Matt and Reddy, Tyler and Cournapeau, David and
            Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and
            Bright, Jonathan and {van der Walt}, St{\'e}fan J. and
            Brett, Matthew and Wilson, Joshua and Millman, K. Jarrod and
            Mayorov, Nikolay and Nelson, Andrew R. J. and Jones, Eric and
            Kern, Robert and Larson, Eric and Carey, C J and
            Polat, {\.I}lhan and Feng, Yu and Moore, Eric W. and
            {VanderPlas}, Jake and Laxalde, Denis and Perktold, Josef and
            Cimrman, Robert and Henriksen, Ian and Quintero, E. A. and
            Harris, Charles R. and Archibald, Anne M. and
            Ribeiro, Ant{\^o}nio H. and Pedregosa, Fabian and
            {van Mulbregt}, Paul and {SciPy 1.0 Contributors}},
  title   = {{{SciPy} 1.0: Fundamental Algorithms for Scientific
            Computing in Python}},
  journal = {Nature Methods},
  year    = {2020},
  volume  = {17},
  pages   = {261--272},
  adsurl  = {https://rdcu.be/b08Wh},
  doi     = {10.1038/s41592-019-0686-2},
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class LabelDistribution(evaluate.Measurement):
    def _info(self):
        return evaluate.MeasurementInfo(
            module_type="measurement",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features({"data": datasets.Value("int32")}),
                datasets.Features({"data": datasets.Value("string")}),
            ],
        )

    def _compute(self, data):
        """Returns the fraction of each label present in the data and the entropy of the distribution."""
        c = Counter(data)
        label_distribution = {"labels": list(c.keys()), "fractions": [f / len(data) for f in c.values()]}
        label_entropy = float(entropy(label_distribution["fractions"]))
        if len(c) > 1:
            label_entropy_normalized = float(label_entropy / np.log(len(c)))
        else:
            label_entropy_normalized = 0.0
        return {
            "label_distribution": label_distribution,
            "label_entropy": label_entropy,
            "label_entropy_normalized": label_entropy_normalized,
        }

# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import evaluate
import datasets
from collections import Counter
import hashlib

logger = evaluate.logging.get_logger(__name__)

_DESCRIPTION = """
Returns the duplicate fraction of duplicate strings in the input.
"""

_KWARGS_DESCRIPTION = """
Args:
    `data`: a list of `str` to be checked for duplicates.

Returns:
    `duplicate_fraction` (`float`) : the fraction of strings that are duplicated.
    `duplicates_list` (`dict`) (optional) : a dictionary containing tuples with the duplicate strings and the number of times they are repeated.

Examples:
    >>> data = ["hello sun","hello moon", "hello sun"]
    >>> duplicates = evaluate.load("text_duplicates")
    >>> results = duplicates.compute(data=data)
    >>> print(results)
    {'duplicate_fraction': 0.33333333333333337}

    >>> data = ["hello sun","hello moon", "hello sun"]
    >>> duplicates = evaluate.load("text_duplicates")
    >>> results =  duplicates.compute(data=data, list_duplicates=True)
    >>> print(results)
    {'duplicate_fraction': 0.33333333333333337, 'duplicates_list': {'hello sun': 2}}
"""

# TODO: Add BibTeX citation
_CITATION = ""


def get_hash(example):
    """Get the hash of a string"""
    return hashlib.md5(example.strip().encode("utf-8")).hexdigest()


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TextDuplicates(evaluate.Measurement):
    """This measurement returns the duplicate strings contained in the input(s)."""

    def _info(self):
        # TODO: Specifies the evaluate.MeasurementInfo object
        return evaluate.MeasurementInfo(
            # This is the description that will appear on the modules page.
            module_type="measurement",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "data": datasets.Value("string"),
                }
            ),
        )

    def _compute(self, data, list_duplicates=False):
        """Returns the duplicates contained in the input data and the number of times they are repeated."""
        if list_duplicates == True:
            logger.warning("This functionality can be memory-intensive for large datasets!")
            n_dedup = len(set([get_hash(d) for d in data]))
            c = Counter(data)
            duplicates = {k: v for k, v in c.items() if v > 1}
            return {"duplicate_fraction": 1 - (n_dedup / len(data)), "duplicates_list": duplicates}
        else:
            n_dedup = len(set([get_hash(d) for d in data]))
            return {"duplicate_fraction": 1 - (n_dedup / len(data))}

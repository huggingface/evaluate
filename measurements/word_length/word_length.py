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

from nltk import word_tokenize
import evaluate
import datasets
from statistics import mean

_DESCRIPTION = """
Returns the length (in words) of the input data.
The tokenizer used is `word_tokenize` from NLTK: https://www.nltk.org/api/nltk.tokenize.html
"""

_KWARGS_DESCRIPTION = """
Args:
    data: a `str` for which the word length is calculated.

Returns:
    'word length' : the number of words in the input string.

Examples:
    >>> data = ["hello world"]
    >>> wordlength = evaluate.load("word_length", type="measurement")
    >>> results = wordlength.compute(data=data)
    >>> print(results)
    {"word length": 2}
"""

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:module,
title = {A great new module},
authors={huggingface, Inc.},
year={2020}
}
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class WordLength(evaluate.EvaluationModule):
    """This measurement returns the number of words in the input string."""

    def _info(self):
        # TODO: Specifies the evaluate.EvaluationModuleInfo object
        return evaluate.EvaluationModuleInfo(
            # This is the description that will appear on the modules page.
            type="measurement",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'data': datasets.Value('string'),
            })
        )

    def _compute(self, data):
        """Returns the word length of the input data"""
        length = len(word_tokenize(data))
        return {"word length": length}

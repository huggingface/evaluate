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
Returns the average length (in terms of the number of words) of the input data.
"""

_KWARGS_DESCRIPTION = """
Args:
    `data`: a list of `str` for which the word length is calculated.
    `tokenizer` (`Callable`) : the approach used for tokenizing `data` (optional).
        The default tokenizer is `word_tokenize` from NLTK: https://www.nltk.org/api/nltk.tokenize.html
        This can be replaced by any function that takes a string as input and returns a list of tokens as output.

Returns:
    `average_word_length` (`float`) : the average number of words in the input list of strings.

Examples:
    >>> data = ["hello world"]
    >>> wordlength = evaluate.load("word_length", type="measurement")
    >>> results = wordlength.compute(data=data)
    >>> print(results)
    {"average_word_length": 2}
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
    """This measurement returns the average number of words in the input string(s)."""

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

    def _compute(self, data, tokenizer=word_tokenize):
        """Returns the average word length of the input data"""
        lengths = [len(tokenizer(d)) for d in data]
        average_length = mean(lengths)
        return {"average_word_length": average_length}

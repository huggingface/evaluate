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
from sklearn.feature_extraction.text import CountVectorizer

_DESCRIPTION = """
Returns the total number of words, and the number of unique words in the input data.
"""

_KWARGS_DESCRIPTION = """
Args:
    `data`: a list of `str` for which the words are counted.
    `max_vocab` (optional): the top number of words to consider (can be specified if dataset is too large)

Returns:
    `total_word_count` (`float`) : the total number of words in the input string(s)
    `unique_words` (`float`) : the number of unique words in the input list of strings.

Examples:
    >>> data = ["hello world and hello moon"]
    >>> wordcount= evaluate.load("word_count")
    >>> results = wordcount.compute(data=data)
    >>> print(results)
    {'total_word_count': 5, 'unique_words': 4}
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
class WordCount(evaluate.EvaluationModule):
    """This measurement returns the total number of words and the number of unique words
     in the input string(s)."""

    def _info(self):
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

    def _compute(self, data, max_vocab = None):
        """Returns the number of unique words in the input data"""
        cvec = CountVectorizer(max_features=max_vocab)
        cvec.fit(data)
        document_matrix = cvec.transform(data)
        word_count = document_matrix.toarray().sum()
        unique_words = document_matrix.shape[1]
        return {"total_word_count": word_count, "unique_words": unique_words}

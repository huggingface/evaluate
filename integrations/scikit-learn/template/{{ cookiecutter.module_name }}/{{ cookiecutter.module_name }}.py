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

from sklearn.metrics import {{ cookiecutter.module_name }}


_CITATION = """\
@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}
"""

_DESCRIPTION = """\
{{ cookiecutter.docstring_first_line }}
"""


_KWARGS_DESCRIPTION = """
    Note: To be consistent with the `evaluate` input conventions the scikit-learn inputs are renamed:
    - `{{ cookiecutter.label_name }}`: `references`
    - `{{ cookiecutter.preds_name }}`: `predictions`
    
    Scikit-learn docstring:
    {{ cookiecutter.docstring }}

"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class {{ cookiecutter.module_class_name }}(evaluate.{{ cookiecutter.module_type | capitalize }}):
    """{{ cookiecutter.docstring_first_line }}"""

    def _info(self):
        return evaluate.{{ cookiecutter.module_type | capitalize }}Info(
            # This is the description that will appear on the modules page.
            module_type="{{ cookiecutter.module_type}}",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features={{ cookiecutter.features }},
            # Homepage of the module for documentation
            homepage="{{ cookiecutter.docs_url }}",
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/scikit-learn/scikit-learn"],
            reference_urls=["https://scikit-learn.org/stable/index.html"]
        )

    def _compute(self, predictions, references, {{ cookiecutter.kwargs }}):
        """Returns the scores"""

        score = {{ cookiecutter.module_name }}({{ cookiecutter.label_name }}=references, {{ cookiecutter.preds_name }}=predictions, {{ cookiecutter.kwargs_input }})

        return {"{{ cookiecutter.module_name }}": score}
# Copyright 2022 The HuggingFace Evaluate Authors
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

# Lint as: python3
""" Comparisons base class."""
import types
from typing import Optional

from datasets.utils.py_utils import copyfunc

from .info import ComparisonInfo
from .naming import camelcase_to_snakecase


class ComparisonInfoMixin:
    """Base class for exposing ComparisonInfo attributes"""

    def __init__(self, info: ComparisonInfo):
        self._comparison_info = info

    @property
    def info(self):
        """:class:`evaluate.ComparisonInfo` object containing all the comparison metadata."""
        return self._comparison_info

    @property
    def name(self) -> str:
        return self._comparison_info.comparison_name

    @property
    def description(self) -> str:
        return self._comparison_info.description

    @property
    def citation(self) -> str:
        return self._comparison_info.citation

    @property
    def inputs_description(self) -> str:
        return self._comparison_info.inputs_description


class Comparison(ComparisonInfoMixin):
    """A Comparison is the base class for comparing two models.

    Args:
        config_name (``str``): Overwritten by comparison loading script.
    """

    def __init__(
        self,
        config_name: Optional[str] = None,
        **kwargs,
    ):
        # prepare info
        self.config_name = config_name or "default"
        info = self._info()
        info.comparison_name = camelcase_to_snakecase(self.__class__.__name__)
        info.config_name = self.config_name
        ComparisonInfoMixin.__init__(self, info)

        self.compute = types.MethodType(copyfunc(self.compute), self)
        self.compute.__func__.__doc__ += self.info.inputs_description

    def __repr__(self):
        return f'Comparison(name: "{self.name}", ' f'usage: """{self.inputs_description}""")'

    def compute(self, *, predictions1=None, predictions2=None, references=None, **kwargs) -> Optional[dict]:
        """Compute the comparison.

        Usage of positional arguments is not allowed to prevent mistakes.

        Args:
            predictions1 (list/array/tensor, optional): Predictions of model 1.
            predictions2 (list/array/tensor, optional): Predictions of model 2.
            references (list/array/tensor, optional): References.
            **kwargs (optional): Keyword arguments that will be forwarded to the metrics :meth:`_compute`
                method (see details in the docstring).

        Return:
            dict or None
        """
        all_kwargs = {"predictions1": predictions1, "predictions2": predictions2, "references": references, **kwargs}
        required_kwargs = ["predictions1", "predictions2", "references"]
        if predictions1 is None and predictions2 is None and references is None:
            missing_kwargs = {k: None for k in required_kwargs if k not in all_kwargs}
            all_kwargs.update(missing_kwargs)
        else:
            missing_inputs = [k for k in required_kwargs if k not in all_kwargs]
            if missing_inputs:
                raise ValueError(
                    f"Metric inputs are missing: {missing_inputs}. All required inputs are {required_kwargs}"
                )
        inputs = {input_name: all_kwargs[input_name] for input_name in required_kwargs}
        compute_kwargs = {k: kwargs[k] for k in kwargs if k not in required_kwargs}

        output = self._compute(**inputs, **compute_kwargs)
        return output

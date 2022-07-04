# Copyright 2022 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
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

from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Lint as: python3
from datasets import Dataset


try:
    from scipy.stats import bootstrap

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from transformers import Pipeline, PreTrainedModel, PreTrainedTokenizer, TFPreTrainedModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from typing_extensions import Literal

from ..loading import load
from ..module import EvaluationModule
from ..utils.logging import get_logger


logger = get_logger(__name__)


class Evaluator(ABC):
    """
    The Evaluator class is the class from which all evaluators inherit. Refer to this class for methods shared across
    different evaluators.
    Base class implementing evaluator operations.
    """

    def __init__(self, task: str, default_metric_name: str = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "If you want to use the `Evaluator` you need `transformers`. Run `pip install evaluate[evaluator]`."
            )
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "If you want to use the `Evaluator` you need `scipy>=1.7.1`. Run `pip install evaluate[evaluator]`."
            )
        self.task = task
        self.default_metric_name = default_metric_name

    @abstractmethod
    def _compute_predictions(self, pipe: "Pipeline", inputs, **predictions_parameters: Dict):
        raise NotImplementedError()

    @staticmethod
    def _compute_confidence_interval(
        predictions,
        references,
        metric: EvaluationModule,
        metric_keys: List[str],
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        A utility function enabling the confidence interval calculation for metrics computed
        by the evaluator based on `scipy`'s `bootstrap` method.
        """
        bootstrap_dict = {}
        for key in metric_keys:
            bs = bootstrap(
                data=(predictions, references),
                statistic=lambda predictions, references: metric.compute(
                    predictions=predictions,
                    references=references,
                )[key],
                paired=True,
                vectorized=False,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
                random_state=random_state,
            )
            bootstrap_dict[key] = {
                "confidence_interval": (bs.confidence_interval.low, bs.confidence_interval.high),
                "standard_error": bs.standard_error,
            }
        return bootstrap_dict

    @abstractmethod
    def compute(
        self,
        model_or_pipeline: Union[str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"] = None,
        data: Union[str, Dataset] = None,
        metric: Union[str, EvaluationModule] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        **compute_parameters: Dict,
    ):
        """
        A core method of the `Evaluator` class, computes the metric value for a pipeline and dataset compatible
        with the task specified by the `Evaluator`.
        """
        raise NotImplementedError()

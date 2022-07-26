# flake8: noqa
# Copyright 2020 The HuggingFace Evaluate Authors and the TensorFlow Datasets Authors.
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
# pylint: enable=line-too-long
# pylint: disable=g-import-not-at-top,g-bad-import-order,wrong-import-position

__version__ = "0.2.1.dev0"

from packaging import version


SCRIPTS_VERSION = "main" if version.parse(__version__).is_devrelease else __version__

del version

from .evaluator import (
    Evaluator,
    ImageClassificationEvaluator,
    QuestionAnsweringEvaluator,
    TextClassificationEvaluator,
    TokenClassificationEvaluator,
    evaluator,
)
from .hub import push_to_hub
from .info import ComparisonInfo, EvaluationModuleInfo, MeasurementInfo, MetricInfo
from .inspect import inspect_evaluation_module, list_evaluation_modules
from .loading import load
from .module import CombinedEvaluations, Comparison, EvaluationModule, Measurement, Metric, combine
from .saving import save
from .utils import *
from .utils import gradio, logging

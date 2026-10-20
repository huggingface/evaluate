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

__version__ = "0.4.7.dev0"

from packaging import version


SCRIPTS_VERSION = "main" if version.parse(__version__).is_devrelease else __version__

del version


import sys
from importlib import import_module
from types import ModuleType

_LAZY_IMPORTS = {
    "AudioClassificationEvaluator": "evaluator",
    "AutomaticSpeechRecognitionEvaluator": "evaluator",
    "CombinedEvaluations": "module",
    "Comparison": "module",
    "ComparisonInfo": "info",
    "EvaluationModule": "module",
    "EvaluationModuleInfo": "info",
    "EvaluationSuite": "evaluation_suite",
    "Evaluator": "evaluator",
    "ImageClassificationEvaluator": "evaluator",
    "Measurement": "module",
    "MeasurementInfo": "info",
    "Metric": "module",
    "MetricInfo": "info",
    "QuestionAnsweringEvaluator": "evaluator",
    "SummarizationEvaluator": "evaluator",
    "Text2TextGenerationEvaluator": "evaluator",
    "TextClassificationEvaluator": "evaluator",
    "TextGenerationEvaluator": "evaluator",
    "TokenClassificationEvaluator": "evaluator",
    "TranslationEvaluator": "evaluator",
    "combine": "module",
    "disable_progress_bar": "utils.logging",
    "enable_progress_bar": "utils.logging",
    "infer_gradio_input_types": "utils.gradio",
    "inspect_evaluation_module": "inspect",
    "is_progress_bar_enabled": "utils.logging",
    "json_to_string_type": "utils.gradio",
    "launch_gradio_widget": "utils.gradio",
    "list_evaluation_modules": "inspect",
    "load": "loading",
    "parse_gradio_data": "utils.gradio",
    "parse_readme": "utils.gradio",
    "parse_test_cases": "utils.gradio",
    "push_to_hub": "hub",
    "save": "saving",
}


_LAZY_MODULES = {
    "config": "config",
    "evaluation_suite": "evaluation_suite",
    "gradio": "utils.gradio",
    "hub": "hub",
    "info": "info",
    "inspect": "inspect",
    "loading": "loading",
    "logging": "utils.logging",
    "module": "module",
    "naming": "naming",
    "saving": "saving",
    "utils": "utils",
}

__all__ = sorted(
    set(_LAZY_IMPORTS) | set(_LAZY_MODULES) | {"evaluator", "SCRIPTS_VERSION"}
)


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = import_module("." + _LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _LAZY_MODULES:
        module = import_module("." + _LAZY_MODULES[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))


class _EvaluateModule(ModuleType):
    def __getattribute__(self, name):
        if name == "evaluator":
            namespace = ModuleType.__getattribute__(self, "__dict__")
            value = namespace.get("evaluator")
            if isinstance(value, ModuleType) or value is None:
                value = import_module(".evaluator", __name__).evaluator
                namespace["evaluator"] = value
            return value
        return ModuleType.__getattribute__(self, name)


sys.modules[__name__].__class__ = _EvaluateModule

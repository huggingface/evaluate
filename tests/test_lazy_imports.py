import subprocess
import sys


def run_python(code):
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_package_import_does_not_load_heavy_dependencies():
    output = run_python(
        "import evaluate, sys; "
        "print(any(name in sys.modules for name in "
        "('datasets', 'pandas', 'pyarrow', 'transformers', 'torch')))"
    )
    assert output == "False"


def test_public_api_exports():
    output = run_python(
        "import evaluate; "
        "assert callable(evaluate.load); "
        "assert callable(evaluate.evaluator); "
        "assert isinstance(evaluate.EvaluationSuite, type); "
        "print('OK')"
    )
    assert output == "OK"


def test_evaluator_remains_callable_after_submodule_import():
    output = run_python(
        "import evaluate, importlib; "
        "importlib.import_module('evaluate.evaluator'); "
        "print(callable(evaluate.evaluator))"
    )
    assert output == "True"


def test_wildcard_import():
    output = run_python(
        "from evaluate import *; "
        "assert callable(load); "
        "assert callable(evaluator); "
        "assert isinstance(EvaluationSuite, type); "
        "print('OK')"
    )
    assert output == "OK"


def test_legacy_public_exports():
    import evaluate

    legacy_names = {
        "AudioClassificationEvaluator",
        "AutomaticSpeechRecognitionEvaluator",
        "CombinedEvaluations",
        "Comparison",
        "ComparisonInfo",
        "EvaluationModule",
        "EvaluationModuleInfo",
        "EvaluationSuite",
        "Evaluator",
        "ImageClassificationEvaluator",
        "Measurement",
        "MeasurementInfo",
        "Metric",
        "MetricInfo",
        "QuestionAnsweringEvaluator",
        "SummarizationEvaluator",
        "Text2TextGenerationEvaluator",
        "TextClassificationEvaluator",
        "TextGenerationEvaluator",
        "TokenClassificationEvaluator",
        "TranslationEvaluator",
        "combine",
        "config",
        "disable_progress_bar",
        "enable_progress_bar",
        "evaluation_suite",
        "evaluator",
        "gradio",
        "hub",
        "infer_gradio_input_types",
        "info",
        "inspect",
        "inspect_evaluation_module",
        "is_progress_bar_enabled",
        "json_to_string_type",
        "launch_gradio_widget",
        "list_evaluation_modules",
        "load",
        "loading",
        "logging",
        "module",
        "naming",
        "parse_gradio_data",
        "parse_readme",
        "parse_test_cases",
        "push_to_hub",
        "save",
        "saving",
        "utils",
    }

    assert all(hasattr(evaluate, name) for name in legacy_names)


def test_wildcard_exports_match_all():
    import evaluate

    namespace = {}
    exec("from evaluate import *", namespace)

    exported = set(namespace) - {"__builtins__"}
    assert exported == set(evaluate.__all__)

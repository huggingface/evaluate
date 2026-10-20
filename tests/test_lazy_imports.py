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

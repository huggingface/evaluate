import importlib
import sys
from unittest.mock import patch

import huggingface_hub
import pytest


CLI_MODULE = "evaluate.commands.evaluate_cli"


@pytest.fixture
def cli_without_hub_repository(monkeypatch):
    """Import the CLI as it would be imported with `huggingface_hub>=1.0.0`.

    `huggingface_hub.Repository` was removed in v1.0.0, so importing it must not be required.
    """
    monkeypatch.delattr(huggingface_hub, "Repository", raising=False)
    monkeypatch.delitem(sys.modules, CLI_MODULE, raising=False)
    return importlib.import_module(CLI_MODULE)


def test_cli_imports_without_hub_repository(cli_without_hub_repository):
    assert hasattr(cli_without_hub_repository, "main")


def test_cli_create_pushes_template_with_git(cli_without_hub_repository, tmp_path):
    evaluate_cli = cli_without_hub_repository
    argv = ["evaluate-cli", "create", "Dummy Metric", "--output_dir", str(tmp_path), "--organization", "dummy_org"]

    with patch.object(evaluate_cli, "create_repo") as create_repo, patch.object(
        evaluate_cli, "cookiecutter"
    ) as cookiecutter, patch.object(evaluate_cli.subprocess, "run") as subprocess_run, patch.object(sys, "argv", argv):
        evaluate_cli.main()

    create_repo.assert_called_once()
    cookiecutter.assert_called_once()

    git_commands = [call.args[0] for call in subprocess_run.call_args_list]
    assert git_commands == [
        ["git", "clone", "https://huggingface.co/spaces/dummy_org/dummy_metric"],
        ["git", "add", "."],
        ["git", "commit", "-m", "add module default template"],
        ["git", "push"],
    ]

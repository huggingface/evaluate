import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union
from abc import ABC, abstractmethod

from datasets import Dataset, DownloadMode, load_dataset
from datasets.utils.version import Version

from ..evaluator import evaluator
from ..loading import evaluation_module_factory
from ..utils.file_utils import DownloadConfig
from ..utils.logging import get_logger


logger = get_logger(__name__)


class Preprocessor(ABC):
    @abstractmethod
    def run(self, dataset: Dataset) -> Dataset:
        pass


@dataclass
class SubTask:
    task_type: str
    data: [Union[str, Dataset]] = None
    subset: Optional[str] = None
    split: Optional[str] = None
    data_preprocessor: Optional[Union[Callable, Preprocessor]] = None
    args_for_task: Optional[dict] = None

    def __post_init__(self):
        if type(self.task_type) is not str:
            raise ValueError(f"'task_type' must be type 'str', got {type(self.task_type)}")
        if type(self.data) not in [Dataset, str]:
            raise ValueError(
                f"'data' must be an already-instantiated Dataset object or type 'str', got {type(self.data)}"
            )
        if self.subset and type(self.subset) is not str:
            raise ValueError(f"'subset' must be type 'str', got {type(self.subset)}")
        if self.split and type(self.split) is not str:
            raise ValueError(f"'split' must be type 'str', got {type(self.split)}")
        if self.data_preprocessor and not callable(self.data_preprocessor):
            raise ValueError(f"'data_preprocessor' must be a Callable', got {self.data_preprocessor}")
        if self.args_for_task and type(self.args_for_task) is not dict:
            raise ValueError(f"'args_for_task' must be type 'dict', got {type(self.args_for_task)}")


def import_main_class(module_path):
    """Import a module at module_path and return the EvaluationSuite class"""
    module = importlib.import_module(module_path)

    module_main_cls = None
    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and obj.__name__ == "Suite":
            if inspect.isabstract(obj):
                continue
            module_main_cls = obj
            break

    return module_main_cls


class EvaluationSuite:
    """
    This class instantiates an evaluation suite made up of multiple tasks, where each task consists of a dataset and
    an associated metric, and runs evaluation on a model or pipeline. Evaluation suites can be a Python script found
    either locally or uploaded as a Space on the Hugging Face Hub.
    Usage:
    ```python
    from evaluate import EvaluationSuite
    suite = EvaluationSuite.load("evaluate/evaluation-suite-ci")
    results = suite.run("lvwerra/distilbert-imdb")
    ```
    """

    def __init__(self, name):
        self.name = name

    @staticmethod
    def load(
        path: str,
        download_mode: Optional[DownloadMode] = None,
        revision: Optional[Union[str, Version]] = None,
        download_config: Optional[DownloadConfig] = None,
    ):
        download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
        evaluation_module = evaluation_module_factory(
            path, module_type=None, revision=revision, download_config=download_config, download_mode=download_mode
        )
        name = Path(path).stem
        evaluation_cls = import_main_class(evaluation_module.module_path)
        evaluation_instance = evaluation_cls(name)

        return evaluation_instance

    def __repr__(self):
        self.tasks = [str(task) for task in self.suite]
        return f'EvaluationSuite name: "{self.name}", ' f"Tasks: {self.tasks})"

    def assert_suite_nonempty(self):
        if not self.suite:
            raise ValueError(
                "No evaluation tasks found. The EvaluationSuite must include at least one SubTask definition."
            )

    def run(
        self, model_or_pipeline: Union[str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"]  # noqa: F821
    ) -> Dict[str, float]:

        self.assert_suite_nonempty()

        results_all = []
        for task in self.suite:

            task_name = task.data

            if task.data_preprocessor:  # task requires extra preprocessing
                ds = load_dataset(task.data, name=task.subset, split=task.split)
                if issubclass(type(task.data_preprocessor), Preprocessor):
                    task.data = task.data_preprocessor.run(ds)
                else:
                    task.data = ds.map(task.data_preprocessor)

            task_evaluator = evaluator(task.task_type)
            args_for_task = task.args_for_task
            args_for_task["model_or_pipeline"] = model_or_pipeline
            args_for_task["data"] = task.data
            args_for_task["subset"] = task.subset
            args_for_task["split"] = task.split
            results = task_evaluator.compute(**args_for_task)

            results["task_name"] = task_name + "/" + task.subset if task.subset else task_name
            results["data_preprocessor"] = str(task.data_preprocessor) if task.data_preprocessor is not None else None
            results_all.append(results)
        return results_all

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from datasets import Dataset, DownloadMode, load_dataset
from datasets.utils.version import Version

from ..evaluator import evaluator
from ..loading import evaluation_module_factory
from ..utils.file_utils import DownloadConfig
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class SubTask:
    task_type: str
    data: Optional[Union[str, Dataset]] = None
    subset: Optional[str] = None
    split: Optional[str] = None
    data_preprocessor: Optional[Callable] = None
    args_for_task: Optional[dict] = None


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
    suite = EvaluationSuite.load('mathemakitten/glue-evaluation-suite')
    results = suite.run("gpt2")
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
        tasks = [task.data + "/" + task.subset if task.subset else task.data for task in self.suite]
        return f'EvaluationSuite name: "{self.name}", ' f"Tasks: {tasks})"

    def run(
        self, model_or_pipeline: Union[str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"]  # noqa: F821
    ) -> Dict[str, float]:

        results_all = {}
        for task in self.suite:

            if task.data_preprocessor:  # task requires extra preprocessing
                ds = load_dataset(task.data, name=task.subset, split=task.split)
                task.data = ds.map(task.data_preprocessor)

            task_evaluator = evaluator(task.task_type)
            args_for_task = task.args_for_task
            args_for_task["model_or_pipeline"] = model_or_pipeline
            args_for_task["data"] = task.data
            args_for_task["subset"] = task.subset
            args_for_task["split"] = task.split
            results = task_evaluator.compute(**args_for_task)

            task_id = task.data + "/" + task.subset if task.subset else task.data
            results_all[task_id] = results
        return results_all

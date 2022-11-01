import json
import os

from evaluate.module import EvaluationModule
from evaluate.evaluator import evaluator
from evaluate.loading import relative_to_absolute_path
from evaluate.utils.file_utils import cached_path, hf_hub_url, is_relative_path
from evaluate.utils.logging import get_logger
from evaluate.config import HF_MODULES_CACHE

from datasets.features import Features, Sequence, Value

from evaluate.module import EvaluationModule, EvaluationModuleInfo, combine

import evaluate

logger = get_logger(__name__)


class EvaluationSuite(EvaluationModule):
    """
    This class instantiates an evaluation suite made up of multiple tasks, where each task consists of a dataset and
    an associated metric, and runs evaluation on a model or pipeline. Evaluation suites can be instantiated from a JSON
    named <module_name>.json found either locally or uploaded as a dataset on the Hugging Face Hub.
    Usage:
    ```python - TODO
    ```
    """

    def _info(self):
        return EvaluationModuleInfo(
            description="dummy metric for tests",
            citation="insert citation here",
            features=Features({"predictions": Value("int64"), "references": Value("int64")}))

    @staticmethod
    def run(suite, model_or_pipeline=None):
        results_all = {}

        suite.setup()
        for task in suite.suite:
            print(task)
            task_evaluator = evaluator("text-classification")
            args_for_task = task.args_for_task
            args_for_task["model_or_pipeline"] = model_or_pipeline
            args_for_task["data"] = task.data
            # args_for_task["subset"] = task.split
            args_for_task["split"] = task.split
            results = task_evaluator.compute(**args_for_task)

            task_id = task.data # TODO FIX THIS + "/" + task.get("name") if task.get("name") else task["data"]
            results_all[task_id] = results
        return results_all


from typing import Optional, Union, Callable
from dataclasses import dataclass
from datasets import Dataset
import dataclasses

@dataclass
class SubTask:
    data: Optional[Union[str, Dataset]] = None
    subset: Optional[str] = None
    split: Optional[str] = None
    data_preprocessor: Optional[Callable] = None
    args_for_task: Optional[dict] = None
    #
    # def __init__(self, **kwargs):
    #     names = set([f.name for f in dataclasses.fields(self)])
    #     for k, v in kwargs.items():
    #         if k in names:
    #             setattr(self, k, v)

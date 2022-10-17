import json
import os

from ..evaluator import evaluator
from ..loading import relative_to_absolute_path
from ..utils.file_utils import cached_path, hf_hub_dataset_url, is_relative_path
from ..utils.logging import get_logger


logger = get_logger(__name__)


def load_evaluation_suite(path):
    return EvaluationSuite(path)


class EvaluationSuite:
    """
    This class instantiates an evaluation suite made up of multiple tasks, where each task consists of a dataset and
    an associated metric, and runs evaluation on a model or pipeline. Evaluation suites can be instantiated from a JSON
    named <module_name>.json found either locally or uploaded as a dataset on the Hugging Face Hub.

    Usage:
    ```python
    >>> from evaluate.evaluation_suite import load_evaluation_suite

    >>> evaluation_suite = load_evaluation_suite('mathemakitten/glue-evaluation_suite')
    >>> results = evaluation_suite.run(model_or_pipeline='gpt2')
    ```
    """

    def __init__(self, path):
        """Instantiates an EvaluationSuite object from a JSON file which will be passed to the Evaluator to run
        evaluation for a model on this collection of tasks."""

        filename = os.path.basename(path)
        if not filename.endswith(".json"):
            filename = filename + ".json"
        combined_path = os.path.join(path, filename)
        if path.endswith(filename):  # Try locally
            if os.path.isfile(path):
                json_filepath = path
                self.config = json.load(open(json_filepath, encoding="utf-8"))
            else:
                raise FileNotFoundError(f"Couldn't find a configuration file at {relative_to_absolute_path(path)}")
        elif os.path.isfile(combined_path):
            self.config = json.load(open(combined_path))
        # Load from a dataset on the Hub
        elif is_relative_path(path) and path.count("/") <= 1:
            json_filepath = cached_path(hf_hub_dataset_url(path=path, name=f"{path.split('/')[-1]}.json"))
            self.config = json.load(open(json_filepath))

        self.tasks = []
        for task_group in self.config["task_groups"]:
            for task in task_group["tasks"]:
                self.tasks.append(task_group["task_type"] + "/" + task["data"])

    def run(self, model_or_pipeline=None):
        results_all = {}
        for task_group in self.config["task_groups"]:
            task_evaluator = evaluator(task_group["task_type"])

            logger.info(f"Running evaluation_suite: {self.config['evaluation_suite_name']} with tasks {self.tasks}")

            for task in task_group["tasks"]:
                logger.info(f"Running task: {task['data']}")

                args_for_task = task["args_for_task"]
                args_for_task["model_or_pipeline"] = model_or_pipeline
                args_for_task["data"] = task["data"]
                args_for_task["subset"] = task.get("name")
                args_for_task["split"] = task.get("split")

                results = task_evaluator.compute(**args_for_task)

                task_id = task["data"] + "/" + task.get("name") if task.get("name") else task["data"]
                results_all[task_id] = results
        return results_all

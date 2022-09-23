import json
import os

from evaluate import evaluator
from evaluate.loading import convert_hf_hub_path, relative_to_absolute_path
from evaluate.utils.file_utils import cached_path, is_relative_path
from evaluate.utils.logging import get_logger


logger = get_logger(__name__)


class Harness:
    """
    This class instantiates an evaluation harness made up of multiple tasks, where each task consists of a dataset and
    an associated metric, and runs evaluation on a model or pipeline. Harnesses can be instantiated from a JSON file
    (harness_config.json) found either locally or in a Space on the Hugging Face Hub.

    Usage:
    ```python
    >>> from evaluate.harness import Harness

    >>> harness = Harness('mathemakitten/sentiment')
    >>> results = harness.run(model_or_pipeline='gpt2')
    ```
    """

    def __init__(self, path):
        """Instantiates a Harness object from a JSON file which will be passed to the Evaluator to run evaluation for a
        model on this collection of tasks."""

        filename = list(filter(lambda x: x, path.replace(os.sep, "/").split("/")))[-1]
        if not filename.endswith(".json"):
            filename = filename + ".json"
        if path.endswith(filename):  # Try locally
            if os.path.isfile(path):
                json_filepath = path
                self.config = json.load(open(json_filepath))
            else:
                raise FileNotFoundError(f"Couldn't find a configuration file at {relative_to_absolute_path(path)}")
        # Load from a space on the Hub
        elif is_relative_path(path) and path.count("/") <= 1:
            json_filepath = cached_path(convert_hf_hub_path(os.path.join(path, "harness_config.json")))
            self.config = json.load(open(json_filepath))

        self.tasks = []
        for task_group in self.config["task_groups"]:
            for task in task_group["tasks"]:
                self.tasks.append(task_group["task_type"] + "/" + task["data"])

    @classmethod
    def from_config(cls, path):
        """
        Instantiates a Harness object from a JSON file which will be passed to the Evaluator to run evaluation for a
        model on this collection of tasks.
        """
        filename = list(filter(lambda x: x, path.replace(os.sep, "/").split("/")))[-1]
        if not filename.endswith(".json"):
            filename = filename + ".json"
        if path.endswith(filename):  # Try locally
            if os.path.isfile(path):
                json_filepath = path
                return json.load(open(json_filepath))
            else:
                raise FileNotFoundError(f"Couldn't find a configuration file at {relative_to_absolute_path(path)}")
        # Load from a space on the Hub
        elif is_relative_path(path) and path.count("/") <= 1:
            json_filepath = cached_path(convert_hf_hub_path(os.path.join(path, "harness_config.json")))
            return json.load(open(json_filepath))
        # TODO: handle caching of the config json file
        # TODO: should we handle "canonical" harnesses?

    def run(self, model_or_pipeline=None):
        results_all = {}
        for task_group in self.config["task_groups"]:
            e = evaluator(task_group["task_type"])

            logger.info(f"Running harness: {self.config['harness_name']} with tasks {self.tasks}")

            for task in task_group["tasks"]:
                logger.info(f"Running task: {task['data']}")

                args_for_task = task["args_for_task"]
                args_for_task["model_or_pipeline"] = model_or_pipeline
                args_for_task["data"] = task["data"]
                args_for_task["subset"] = task.get("name")

                results = e.compute(**args_for_task)

                task_id = task["data"] + "/" + task.get("name") if task.get("name") else task["data"]
                results_all[task_id] = results
        return results_all

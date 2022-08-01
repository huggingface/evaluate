import json
import os

from datasets import Dataset, load_dataset

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

    >>> harness_config = Harness.from_json('mathemakitten/sentiment')
    >>> harness = Harness(harness_config)
    >>> results = harness.run()
    ```
    """

    def __init__(self, config):
        self.config = config
        self.tasks = self.config["tasks"]

    @classmethod
    def from_json(cls, path):
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

    def run(self):
        e = evaluator(self.config["task_type"])

        logger.info(f"Running harness: {self.config['harness_name']} with tasks {self.tasks}")

        results_all = {}
        for task in self.tasks:
            logger.info(f"Running task: {task['name']}")

            data = Dataset.from_dict(load_dataset(task['data'])["test"][:])

            args_for_task = task["args_for_task"]
            args_for_task["model_or_pipeline"] = self.config[
                "model_or_pipeline"
            ]  # TODO: allow for passing in arbitrary callable
            args_for_task["data"] = data

            results = e.compute(**args_for_task)

            results_all[task["name"]] = results
        return results_all

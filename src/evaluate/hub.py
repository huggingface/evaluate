from typing import Dict

import requests
from datasets.utils.metadata import known_task_ids
from huggingface_hub import dataset_info, model_info
from huggingface_hub.repocard import metadata_update

from .utils.logging import get_logger


logger = get_logger(__name__)


def get_allowed_tasks(tasks_dict):
    return list(tasks_dict.keys())


def push_to_hub(
    model_id: str,
    task_type: str,
    dataset_type: str,
    dataset_name: str,
    metric_type: str,
    metric_name: str,
    metric_value: float,
    task_name: str = None,
    dataset_config: str = None,
    dataset_split: str = None,
    dataset_revision: str = None,
    dataset_args: Dict[str, int] = None,
    metric_config: str = None,
    metric_args: Dict[str, int] = None,
    overwrite: bool = False,
):
    r"""
    Pushes the result of a metric to the metadata of a model repository in the Hub.

    Args:
        model_id (``str``): Model id from https://hf.co/models.
        task_type (``str``): Task id, refer to
            https://github.com/huggingface/datasets/blob/master/src/datasets/utils/resources/tasks.json for allowed values.
        dataset_type (``str``): Dataset id from https://hf.co/datasets.
        dataset_name (``str``): Pretty name for the dataset.
        metric_type (``str``): Metric id from https://hf.co/metrics.
        metric_name (``str``): Pretty name for the metric.
        metric_value (``float``): Computed metric value.
        task_name (``str``, optional): Pretty name for the task.
        dataset_config (``str``, optional): Dataset configuration used in datasets.load_dataset().
            See huggingface/datasets docs for more info: https://huggingface.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset.name
        dataset_split (``str``, optional): Name of split used for metric computation.
        dataset_revision (``str``, optional): Git hash for the specific version of the dataset.
        dataset_args (``dict[str, int]``, optional): Additional arguments passed to datasets.load_dataset().
        metric_config (``str``, optional): Configuration for the metric (e.g. the GLUE metric has a configuration for each subset)
        metric_args (``dict[str, int]``, optional): Arguments passed during Metric.compute().
        overwrite (``bool``, optional, defaults to `False`): If set to `True` an existing metric field can be overwritten, otherwise
             attempting to overwrite any existing fields will cause an error.

    Example:

    ```python
    >>> push_to_hub(
    ...     model_id="huggingface/gpt2-wikitext2",
    ...     metric_value=0.5
    ...     metric_type="bleu",
    ...     metric_name="BLEU",
    ...     dataset_name="WikiText",
    ...     dataset_type="wikitext",
    ...     dataset_split="test",
    ...     task_type="text-generation",
    ...     task_name="Text Generation"
    ... )
    ```"""
    tasks = get_allowed_tasks(known_task_ids)

    if task_type not in tasks:
        raise ValueError(f"Task type not supported. Task has to be one of {tasks}")

    try:
        dataset_info(dataset_type)
    except requests.exceptions.HTTPError:
        logger.warning(f"Dataset {dataset_type} not found on the Hub at hf.co/datasets/{dataset_type}")

    try:
        model_info(model_id)
    except requests.exceptions.HTTPError:
        raise ValueError(f"Model {model_id} not found on the Hub at hf.co/{model_id}")

    result = {
        "task": {
            "type": task_type,
        },
        "dataset": {
            "type": dataset_type,
            "name": dataset_name,
        },
        "metrics": [
            {
                "type": metric_type,
                "value": metric_value,
            },
        ],
    }

    if dataset_config is not None:
        result["dataset"]["config"] = dataset_config
    if dataset_split is not None:
        result["dataset"]["split"] = dataset_split
    if dataset_revision is not None:
        result["dataset"]["revision"] = dataset_revision
    if dataset_args is not None:
        result["dataset"]["args"] = dataset_args

    if task_name is not None:
        result["task"]["name"] = task_name

    if metric_name is not None:
        result["metrics"][0]["name"] = metric_name
    if metric_config is not None:
        result["metrics"][0]["config"] = metric_config
    if metric_args is not None:
        result["metrics"][0]["args"] = metric_args

    metadata = {"model-index": [{"results": [result]}]}

    return metadata_update(repo_id=model_id, metadata=metadata, overwrite=overwrite)

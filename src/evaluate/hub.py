from huggingface_hub.repocard import metadata_update


def push_to_hub(
    to: str,
    task_type: str,
    metric_value: float,
    metric_type: str,
    dataset_name: str,
    dataset_type: str,
    task_name: str = None,
    dataset_config: str = None,
    dataset_split: str = None,
    dataset_revision: str = None,
    dataset_args: dict[str, int] = None,
    metric_name: str = None,
    metric_config: str = None,
    metric_args: dict[str, int] = None,
):
    """
    TODO: Add documentation
    """

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

    if dataset_config:
        result["dataset"]["config"] = dataset_config
    if dataset_split:
        result["dataset"]["split"] = dataset_split
    if dataset_revision:
        result["dataset"]["revision"] = dataset_revision
    if dataset_args:
        result["dataset"]["args"] = dataset_args

    if task_name:
        result["task"]["name"] = task_name

    if metric_name:
        result["metrics"][0]["name"] = metric_name
    if metric_config:
        result["metrics"][0]["config"] = metric_config
    if metric_args:
        result["metrics"][0]["args"] = metric_args

    metadata = {"model-index": [{"results": [result]}]}

    # TODO: Do we also want to add to the 'metrics' outside of model-index?

    return metadata_update(repo_id=to, metadata=metadata, overwrite=True)

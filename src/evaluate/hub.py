from huggingface_hub.repocard import metadata_update


def push_to_hub(
    repo_id: str,
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
    overwrite: bool = False,
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

    # TODO: Do we also want to add to the 'metrics' outside of model-index?

    return metadata_update(repo_id=repo_id, metadata=metadata, overwrite=overwrite)

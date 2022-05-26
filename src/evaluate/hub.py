from huggingface_hub.repocard import metadata_update


def push_to_hub(
    to: str,
    metric_value: str,
    metric_type: str,
    dataset_name: str,
    dataset_type: str,
    task_type: str,
    **kwargs,
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

    if kwargs.get("task_name"):
        result["task"]["name"] = kwargs.get("task_name")
    # TODO: Add additional (optional) elements...

    metadata = {"model-index": [{"results": [result]}]}

    return metadata_update(repo_id=to, metadata=metadata, overwrite=True)

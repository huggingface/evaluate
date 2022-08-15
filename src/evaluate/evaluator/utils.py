from datasets import Dataset, get_dataset_split_names


class DatasetColumn(list):
    """Helper class to avoid loading a dataset column into memory when accessing it."""

    def __init__(self, dataset: Dataset, key: str):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][self.key]

    def __iter__(self):
        return (self.dataset[i][self.key] for i in range(len(self)))


def choose_split(data):
    available_splits = get_dataset_split_names(data)
    preferred_split_order = [
        "test",
        "testing",
        "eval",
        "evaluation",
        "validation",
        "val",
        "valid",
        "dev",
        "train",
        "training",
    ]
    for split in preferred_split_order:
        if split in available_splits:
            return split
    raise ValueError("No dataset split defined! Pass an explicit value to the `data_split` kwarg.")

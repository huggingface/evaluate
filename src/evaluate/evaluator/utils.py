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
    raise ValueError("No dataset split defined! Pass an explicit value to the `split` kwarg.")


class DatasetColumnPair(list):
    """Helper class to avoid loading two dataset columns into memory when accessing it."""

    def __init__(
        self,
        dataset: Dataset,
        first_col: str,
        second_col: str,
        first_key: str,
        second_key: str,
    ):
        """
        Args:
            dataset (Dataset): dataset to build an iterator on
            first_col (str): first column name to use in the dataset
            second_col (str): second column name to use in the dataset
            first_key (str): key name used for the first column in the returned dictionary
            second_key (str): key name used for the second column in the returned dictionary
        """
        self.dataset = dataset

        self.first_col = first_col
        self.second_col = second_col

        self.first_key = first_key
        self.second_key = second_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return {
            self.first_key: self.dataset[i][self.first_col],
            self.second_key: self.dataset[i][self.second_col] if self.second_col else None,
        }

    def __iter__(self):
        return (
            {
                self.first_key: self.dataset[i][self.first_col],
                self.second_key: self.dataset[i][self.second_col] if self.second_col else None,
            }
            for i in range(len(self))
        )

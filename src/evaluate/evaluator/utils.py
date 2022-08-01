from datasets import Dataset


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


class DatasetColumnPair(list):
    """Helper class to avoid loading a dataset column into memory when accessing it."""

    def __init__(self, dataset: Dataset, first_key: str, second_key: str):
        self.dataset = dataset
        self.first_key = first_key
        self.second_key = second_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return {"text": self.dataset[i][self.first_key], "text_pair": self.dataset[i][self.second_key] if self.second_key else None}

    def __iter__(self):
        return (
            {"text": self.dataset[i][self.first_key], "text_pair": self.dataset[i][self.second_key] if self.second_key else None}
            for i in range(len(self))
        )

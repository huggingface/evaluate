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

    def __init__(self, dataset: Dataset, key: str, key2: str):
        self.dataset = dataset
        self.key = key
        self.key2 = key2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return {"text": self.dataset[i][self.key], "text_pair": self.dataset[i][self.key2] if self.key2 else None}

    def __iter__(self):
        return (
            {"text": self.dataset[i][self.key], "text_pair": self.dataset[i][self.key2] if self.key2 else None}
            for i in range(len(self))
        )

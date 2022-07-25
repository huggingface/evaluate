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

from datasets import Dataset

from evaluate.evaluator.utils import DatasetColumn, DatasetColumnPair


def test_dataset_column_repr():
    dataset = Dataset.from_dict({"text": ["a", "b", "c"]})
    column = DatasetColumn(dataset, "text")

    assert repr(column) == "DatasetColumn(key='text', len=3)"


def test_dataset_column_pair_repr():
    dataset = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 0]})
    column_pair = DatasetColumnPair(dataset, "text", "label", "prediction", "reference")

    assert repr(column_pair) == (
        "DatasetColumnPair(first_col='text', second_col='label', "
        "first_key='prediction', second_key='reference', len=3)"
    )

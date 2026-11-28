import pytest
from datasets import Dataset

from evaluate.evaluator.utils import DatasetColumnPair


@pytest.mark.parametrize("second_column", ["hypothesis", None])
def test_dataset_column_pair_formats_each_row_once(second_column):
    formatted_rows = []

    def transform(batch):
        formatted_rows.extend(batch["premise"])
        return batch

    data = Dataset.from_dict({"premise": ["first", "second"], "hypothesis": ["one", "two"]})
    inputs = DatasetColumnPair(data.with_transform(transform), "premise", second_column, "text", "text_pair")
    expected = [
        {"text": "first", "text_pair": "one" if second_column else None},
        {"text": "second", "text_pair": "two" if second_column else None},
    ]

    assert inputs[-1] == expected[1]
    assert formatted_rows == ["second"]

    formatted_rows.clear()
    assert list(inputs) == expected
    assert formatted_rows == ["first", "second"]

from pathlib import Path

import pytest
from datasets import Dataset

import evaluate
from evaluate.evaluation_suite import SubTask


class LocalClassificationPipeline:
    task = "text-classification"

    def __init__(self, raises_once):
        self.raises_once = raises_once
        self.inputs = []

    def __call__(self, inputs, **kwargs):
        texts = [row["text"] for row in inputs]
        self.inputs.append(texts)
        if self.raises_once:
            self.raises_once = False
            raise RuntimeError("pipeline failed")
        return [{"label": int(text == "1 processed")} for text in texts]


@pytest.mark.parametrize("in_memory", [False, True])
@pytest.mark.parametrize("raises_once", [False, True])
def test_suite_preprocessing_preserves_source_data(tmp_path, in_memory, raises_once):
    raw = Dataset.from_dict({"text": ["0", "1"], "label": [0, 1]})
    if in_memory:
        data = raw
        split = None
    else:
        directory = tmp_path / "dataset"
        directory.mkdir()
        raw.to_csv(directory / "train.csv", index=False)
        data = str(directory)
        split = "train"
    metric = evaluate.load(str(Path(__file__).parents[1] / "metrics" / "accuracy"), keep_in_memory=True)
    task = SubTask(
        "text-classification",
        data=data,
        split=split,
        data_preprocessor=lambda row: {"text": f"{row['text']} processed"},
        args_for_task={"metric": metric, "device": -1},
    )
    suite = evaluate.EvaluationSuite("local")
    suite.suite = [task]
    pipeline = LocalClassificationPipeline(raises_once)

    if raises_once:
        with pytest.raises(RuntimeError, match="pipeline failed"):
            suite.run(pipeline)
    else:
        assert suite.run(pipeline)[0]["accuracy"] == 1.0
    assert task.data is data

    result = suite.run(pipeline)[0]
    assert result["accuracy"] == 1.0
    assert result["task_name"] is data
    assert task.data is data
    assert raw["text"] == ["0", "1"]
    assert pipeline.inputs == [["0 processed", "1 processed"], ["0 processed", "1 processed"]]

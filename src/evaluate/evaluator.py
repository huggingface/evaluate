from typing import Dict, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_metric
from datasets.metric import Metric
from transformers import Pipeline, pipeline


class Evaluator:
    def __init__(
        self,
        pipe: Union[str, Pipeline],
        data: Union[str, DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
        metric: Union[str, Metric],
        col_mapping: Optional[Dict[str, str]] = None,
        label_mapping: Optional[Dict[str, str]] = None,
    ):
        self.pipe = pipeline(pipe) if isinstance(pipe, str) else pipe
        self.data = load_dataset(data) if isinstance(data, str) else data
        self.metric = load_metric(metric) if isinstance(metric, str) else metric
        self.label_mapping = label_mapping
        if col_mapping is not None:
            self.data = data.rename_columns(col_mapping)

    def get_predictions(self, baseline_pipe: Optional[Pipeline] = None):
        predictions = (
            baeline_pipe(self.data["inputs"], truncation=True)
            if baseline_pipe is not None
            else self.pipe(self.data["inputs"], truncation=True)
        )
        return [
            self.label_mapping[element["label"]] if self.label_mapping is not None else element["label"]
            for element in predictions
        ]

    def get_references(self):
        return self.data["references"]

    def get_confidence_intercal(self):
        pass

    def get_bootstrap_p_value(self):
        pass

    def compute(self):
        predictions = self.get_predictions()
        references = self.get_references()
        result = self.metric.compute(predictions=predictions, references=references)
        return result

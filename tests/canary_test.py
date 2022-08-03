""" These canary datasets are used to test loaded models against a reference to understand if these models are the same
by checking model predictions on some known canary datasets. """
from datasets import Dataset, load_dataset


class CanaryDataset:
    def __init__(self, evaluator_task, input_columns, label_column):
        if evaluator_task in ["text-classification", "sentiment-analysis"]:
            data = load_dataset("sst2")["test"][:2]
            data[input_columns] = data.pop("sentence")
            data[label_column] = data.pop("label")
        elif evaluator_task == 'image-classification':
            data = load_dataset("beans", split="test")[:2]
            data[input_columns] = data.pop("image")
            data[label_column] = data.pop("labels")
        else:
            raise ValueError(f"Canary dataset not implemented for this Evaluator task: {evaluator_task}!")
        # TODO: fill in canaries for other evaluators
        self.data = Dataset.from_dict(data)

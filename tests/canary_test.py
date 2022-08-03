""" These canary datasets are used to test loaded models against a reference to understand if these models are the same
by checking model predictions on some known canary datasets. """
from datasets import Dataset


class CanaryDataset:
    def __init__(self, evaluator_task, input_column, label_column):
        if evaluator_task in ["text-classification", "sentiment-analysis"]:
            canary_examples = {
                "idx": [0, 1, 2],
                input_column: [
                    "blue whales are so big",
                    "the ocean is so vast",
                    "shrimp are the most untrustworthy ocean creature",
                ],
                label_column: [1, 1, 0],
            }
        # TODO: fill in canaries for other evaluators
        self.data = Dataset.from_dict(canary_examples)

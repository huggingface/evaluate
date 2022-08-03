""" These canary datasets are used to test loaded models against a reference to understand if these models are the same
by checking model predictions on some known canary datasets. """
from datasets import Dataset, load_dataset


class CanaryDataset:
    def __init__(self, evaluator_task, input_columns, label_column):
        if evaluator_task in ["text-classification", "sentiment-analysis"]:
            data = {
                "idx": [0, 1, 2],
                input_columns: [
                    "blue whales are so big",
                    "the ocean is so vast",
                    "shrimp are the most untrustworthy ocean creature",
                ],
                label_column: [1, 1, 0],
            }
        elif evaluator_task == "image-classification":
            data = load_dataset("beans", split="test")[:2]
            data[input_columns] = data.pop("image")
            data[label_column] = data.pop("labels")
        elif evaluator_task == "question-answering":
            data = {
                "id": ["001", "002"],
                "title": ["Ocean animals", "Whales"],
                "context": [
                    "A whale is bigger than every fish on the planet! it is bigger than the moon!!! whale!!",
                    "finding a whale is like finding a needle in a haystack, except the needle is a whale and the haystack is the ocean",
                ],
                "question": ["Is a whale bigger than the moon??", "If the ocean is a haystack, what is a whale??"],
                "answers": [
                    {"text": ["yes", "yes", "yes"], "answer_start": [3, 3, 3]},
                    {"text": ["A needle", "A needle", "A needle"], "answer_start": [7, 7, 7]},
                ],
            }
        else:
            raise ValueError(f"Canary dataset not implemented for this Evaluator task: {evaluator_task}!")
        # TODO: fill in canaries for other evaluators
        self.data = Dataset.from_dict(data)

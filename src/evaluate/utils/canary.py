""" These canary datasets are used to test loaded models against a reference to understand if these models are the same
by checking model predictions on some known canary datasets. """
import datasets
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
            self.data = Dataset.from_dict(data)
        elif evaluator_task == "image-classification":
            data = load_dataset("beans", split="test")[:2]
            data[input_columns] = data.pop("image")
            data[label_column] = data.pop("labels")
            self.data = Dataset.from_dict(data)
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
            self.data = Dataset.from_dict(data)
        elif evaluator_task == "token-classification":
            data = {
                "id": ["0", "1"],
                "tokens": [
                    ["NOBODY", "SENT", "THE", "BELUGA", "WHALE", "A", "BIRTHDAY", "CARD", "LAST", "WEEK", "."],
                    ["SHRIMP", "CAKE"],
                ],
                "pos_tags": [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5]],
                "chunk_tags": [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], [5, 5]],
                "ner_tags": [[0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [5, 0]],
            }
            self.data = Dataset.from_dict(
                data,
                features=datasets.Features(
                    {
                        "id": datasets.Value("string"),
                        "tokens": datasets.Sequence(datasets.Value("string")),
                        "pos_tags": datasets.Sequence(
                            datasets.features.ClassLabel(
                                names=[
                                    '"',
                                    "''",
                                    "#",
                                    "$",
                                    "(",
                                    ")",
                                ]
                            )
                        ),
                        "chunk_tags": datasets.Sequence(
                            datasets.features.ClassLabel(
                                names=[
                                    "O",
                                    "B-ADJP",
                                    "I-ADJP",
                                    "B-ADVP",
                                    "I-ADVP",
                                    "B-CONJP",
                                    "I-CONJP",
                                    "B-INTJ",
                                ]
                            )
                        ),
                        "ner_tags": datasets.Sequence(
                            datasets.features.ClassLabel(
                                names=[
                                    "O",
                                    "B-PER",
                                    "I-PER",
                                    "B-ORG",
                                    "I-ORG",
                                    "B-LOC",
                                ]
                            )
                        ),
                    }
                ),
            )
        else:
            raise ValueError(f"Canary dataset not implemented for this Evaluator task: {evaluator_task}!")

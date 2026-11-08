# Copyright 2022 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from evaluate import Metric, MetricInfo, TokenClassificationEvaluator


@pytest.fixture
def whitespace_pipeline():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "New": 1, "York": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")

    class WhitespacePipeline:
        task = "token-classification"

        def __call__(self, inputs, **kwargs):
            predictions = []
            for text in inputs:
                encoded = fast_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
                predictions.append(
                    [
                        {"start": start, "entity": {0: "O", 1: "B-LOC", 2: "I-LOC"}[token_id]}
                        for token_id, (start, _) in zip(encoded["input_ids"], encoded["offset_mapping"])
                    ]
                )
            return predictions

    return WhitespacePipeline()


class TokenAccuracy(Metric):
    def _info(self):
        return MetricInfo(
            description="Token accuracy for local evaluator tests.",
            citation="",
            inputs_description="Aligned sequences of token labels.",
            features=Features({"predictions": Sequence(Value("string")), "references": Sequence(Value("string"))}),
        )

    def _compute(self, predictions, references):
        self.predictions = predictions
        assert len(predictions) == len(references)
        assert all(len(prediction) == len(reference) for prediction, reference in zip(predictions, references))
        correct = sum(
            prediction == reference
            for predicted_row, reference_row in zip(predictions, references)
            for prediction, reference in zip(predicted_row, reference_row)
        )
        return {"accuracy": correct / sum(len(reference) for reference in references)}


@pytest.mark.parametrize(
    "words,expected",
    [
        (["New", "York"], ["B-LOC", "I-LOC"]),
        (["New", " ", "York"], ["B-LOC", "O", "I-LOC"]),
        ([" ", "New", "York"], ["O", "B-LOC", "I-LOC"]),
        (["New", "York", " "], ["B-LOC", "I-LOC", "O"]),
        (["New", "York", " ", "\t"], ["B-LOC", "I-LOC", "O", "O"]),
        (["   "], ["O"]),
        (["", ""], ["O", "O"]),
        ([], []),
    ],
)
def test_predictions_processor_with_tokenizer_omitted_words(whitespace_pipeline, words, expected):
    predictions = whitespace_pipeline([" ".join(words)])
    result = TokenClassificationEvaluator().predictions_processor(predictions, [words], " ")
    assert result == {"predictions": [expected]}


@pytest.mark.parametrize(
    "words,predictions,expected",
    [
        (["New", " "], [{"start": 0, "entity": "B-LOC"}, {"start": 1, "entity": "I-LOC"}], ["B-LOC", "O"]),
        (
            ["New", "York", " "],
            [
                {"start": 0, "entity": "B-LOC"},
                {"start": 1, "entity": "I-LOC"},
                {"start": 4, "entity": "I-LOC"},
                {"start": 5, "entity": "I-LOC"},
            ],
            ["B-LOC", "I-LOC", "O"],
        ),
    ],
)
def test_predictions_processor_exhausts_subtokens_before_omitted_tail(words, predictions, expected):
    result = TokenClassificationEvaluator().predictions_processor([predictions], [words], " ")
    assert result == {"predictions": [expected]}


def test_predictions_processor_resets_cursor_after_empty_prediction():
    result = TokenClassificationEvaluator().predictions_processor(
        [[], [{"start": 0, "entity": "B-LOC"}]], [[" "], ["New"]], " "
    )
    assert result == {"predictions": [["O"], ["B-LOC"]]}


def test_compute_with_tokenizer_omitted_words(whitespace_pipeline, tmp_path):
    data = Dataset.from_dict(
        {
            "tokens": [["New", "York", " "], ["   "], ["New", " ", "York"], []],
            "ner_tags": [[1, 2, 0], [0], [1, 0, 2], []],
        },
        features=Features(
            {
                "tokens": Sequence(Value("string")),
                "ner_tags": Sequence(ClassLabel(names=["O", "B-LOC", "I-LOC"])),
            }
        ),
    )
    metric = TokenAccuracy(cache_dir=str(tmp_path))
    result = TokenClassificationEvaluator().compute(
        model_or_pipeline=whitespace_pipeline, data=data, metric=metric, device=-1
    )
    assert result["accuracy"] == 1.0
    assert metric.predictions == [["B-LOC", "I-LOC", "O"], ["O"], ["B-LOC", "O", "I-LOC"], []]

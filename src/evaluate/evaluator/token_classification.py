# Copyright 2022 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
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

from abc import ABC, abstractmethod
from functools import partial
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Lint as: python3
from datasets import ClassLabel, Dataset, Sequence, load_dataset


try:
    from transformers import Pipeline, PreTrainedModel, PreTrainedTokenizer, TFPreTrainedModel, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from typing_extensions import Literal

from ..loading import load
from ..module import EvaluationModule
from ..utils.logging import get_logger
from .base import Evaluator


logger = get_logger(__name__)


class TokenClassificationEvaluator(Evaluator):
    """
    Token classification evaluator.

    This token classification evaluator can currently be loaded from [`evaluator`] using the default task name
    `token-classification`.

    Methods in this class assume a data format compatible with the [`TokenClassificationPipeline`].

    In particular, the following cases are not handled:
    * models not splitting tokens by space (e.g. where `"he oh"` can be a token).
    * datasets as https://huggingface.co/datasets/msra_ner , where tokens are provided ideogram by ideogram, and where a tokenizer may map several ideograms to a single token.
    * datasets not providing the input and reference columns as a list of "words" as the conll2003 dataset.
    """

    def __init__(self, task="token-classification", default_metric_name=None):
        super().__init__(task, default_metric_name=default_metric_name)

    # Tokenize all texts and align the labels with them.
    def _tokenize_and_align_labels(self, examples, tokenizer, input_column):
        tokenized_inputs = tokenizer(
            examples[input_column],
            padding=False,
            truncation=True,
            max_length=tokenizer.model_max_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        """
        labels = []
        for i, label in enumerate(examples[label_column]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        """
        return tokenized_inputs

    def _compute_predictions(
        self, pipe: "Pipeline", inputs, label_mapping: Optional[Dict] = None, join_by: str = " "
    ) -> List[Number]:
        all_preds = []

        for data in inputs:
            inp = join_by.join(data)
            res = pipe(inp)

            # BatchEncoding.word_ids may be wrong (as we joined words with " "), so let's populate it ourselves
            token_to_word_id = []
            for j, word in enumerate(data):
                preprocessed_inputs = pipe.preprocess(word)
                n_tokens = len([k for k in preprocessed_inputs.word_ids(0) if k != None])  # exclude None
                token_to_word_id.extend([j] * n_tokens)

            # the pipeline may give as output labeled tokens that are part of the same word, keep track
            # of the indexing to match the true labels on words
            index_tokens_word_start = []

            for j, word_index in enumerate(token_to_word_id):
                if j == 0:
                    index_tokens_word_start.append(j)
                elif word_index != token_to_word_id[j - 1]:
                    index_tokens_word_start.append(j)

            # keep only predictions that correspond to the beginning of a word
            if label_mapping:
                preds = [label_mapping[res[index]["entity"]] for index in index_tokens_word_start]
            else:
                preds = [res[index]["entity"] for index in index_tokens_word_start]

            all_preds.append(preds)

        return all_preds

    """
    def _compute_predictions(
        self, pipe: "Pipeline", inputs, label_mapping: Optional[Dict] = None, join_by: str = " "
    ) -> List[Number]:
        all_preds = []

        for data in inputs:
            inp = join_by.join(data)
            res = pipe(inp)

            # BatchEncoding.word_ids may be wrong (as we joined words with " "), so let's populate it ourselves
            token_to_word_id = []
            for j, word in enumerate(data):
                preprocessed_inputs = pipe.preprocess(word)
                n_tokens = len([k for k in preprocessed_inputs.word_ids(0) if k != None])  # exclude None
                token_to_word_id.extend([j] * n_tokens)

            # the pipeline may give as output labeled tokens that are part of the same word, keep track
            # of the indexing to match the true labels on words
            index_tokens_word_start = []

            for j, word_index in enumerate(token_to_word_id):
                if j == 0:
                    index_tokens_word_start.append(j)
                elif word_index != token_to_word_id[j - 1]:
                    index_tokens_word_start.append(j)

            # keep only predictions that correspond to the beginning of a word
            if label_mapping:
                preds = [label_mapping[res[index]["entity"]] for index in index_tokens_word_start]
            else:
                preds = [res[index]["entity"] for index in index_tokens_word_start]

            all_preds.append(preds)

        return all_preds
    """

    def compute(
        self,
        model_or_pipeline: Union[str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"] = None,
        data: Union[str, Dataset] = None,
        metric: Union[str, EvaluationModule] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        random_state: Optional[int] = None,
        input_column: str = "tokens",
        label_column: str = "ner_tags",
        label_mapping: Optional[Dict] = None,
        join_by: Optional[str] = " ",
    ) -> Tuple[Dict[str, float], Any]:
        """
        Compute the metric for a given pipeline and dataset combination.

        Args:
            model_or_pipeline (`str` or `Pipeline` or `Callable` or `PreTrainedModel` or `TFPreTrainedModel`,
            defaults to `None`):
                If the argument in not specified, we initialize the default pipeline for the task (in this case
                `token-classification`). If the argument is of the type `str` or
                is a model instance, we use it to initialize a new `Pipeline` with the given model. Otherwise we assume the
                argument specifies a pre-initialized pipeline.
            data (`str` or `Dataset`, defaults to `None):
                Specifies the dataset we will run evaluation on. If it is of type `str`, we treat it as the dataset
                name, and load it. Otherwise we assume it represents a pre-loaded dataset.
            metric (`str` or `EvaluationModule`, defaults to `None`"
                Specifies the metric we use in evaluator. If it is of type `str`, we treat it as the metric name, and
                load it. Otherwise we assume it represents a pre-loaded metric.
            tokenizer: (`str` or `PreTrainedTokenizer`, *optional*, defaults to `None`):
                Argument can be used to overwrite a default tokenizer if `model_or_pipeline` represents a model for
                which we build a pipeline. If `model_or_pipeline` is `None` or a pre-initialized pipeline, we ignore
                this argument.
            strategy: (`Literal["simple", "bootstrap"]`, defaults to "simple"):
                specifies the evaluation strategy. Possible values are:

                - `"simple"` - we evaluate the metric and return the scores.
                - `"bootstrap"` - on top of computing the metric scores, we calculate the confidence interval for each
                of the returned metric keys, using `scipy`'s `bootstrap` method
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html.
            confidence_level (`float`, defaults to `0.95`):
                The `confidence_level` value passed to `bootstrap` if `"bootstrap"` strategy is chosen.
            n_resamples (`int`, defaults to `9999`):
                The `n_resamples` value passed to `bootstrap` if `"bootstrap"` strategy is chosen.
            random_state (`int`, *optional*, defaults to `None`):
                The `random_state` value passed to `bootstrap` if `"bootstrap"` strategy is chosen. Useful for
                debugging.
            input_column (`str`, defaults to `"tokens"`):
                the name of the column containing the tokens feature in the dataset specified by `data`.
            label_column (`str`, defaults to `"label"`):
                the name of the column containing the labels in the dataset specified by `data`.
            label_mapping (`Dict`, *optional*, defaults to `None`):
                We want to map class labels defined by the model in the pipeline to values consistent with those
                defined in the `label_column` of the `data` dataset.
            join_by (`str`, *optional*, defaults to `" "`):
                This evaluator supports dataset whose input column is a list of words. This parameter specifies how to join
                words to generate a string input. This is especially useful for languages that do not separate words by a space.

        Return:
            A `Dict`. The keys represent metric keys calculated for the `metric` spefied in function arguments. For the
            `"simple"` strategy, the value is the metric score. For the `"bootstrap"` strategy, the value is a `Dict`
            containing the score, the confidence interval and the standard error calculated for each metric key.
        """
        # Prepare data.
        if data is None:
            raise ValueError(
                "Please specify a valid `data` object - either a `str` with a name or a `Dataset` object."
            )
        data = load_dataset(data) if isinstance(data, str) else data
        if input_column not in data.column_names:
            raise ValueError(
                f"Invalid `input_column` {input_column} specified. The dataset contains the following columns: {data.column_names}."
            )
        if label_column not in data.column_names:
            raise ValueError(
                f"Invalid `label_column` {label_column} specified. The dataset contains the following columns: {data.column_names}."
            )

        if not isinstance(data.features[input_column], Sequence):
            raise NotImplementedError("Only datasets with the input column as a list of word is supported.")

        kwargs = {
            "ignore_labels": [],  # do not ignore "O"
        }
        # Prepare pipeline.
        if (
            isinstance(model_or_pipeline, PreTrainedModel)
            or isinstance(model_or_pipeline, TFPreTrainedModel)
            or isinstance(model_or_pipeline, str)
        ):
            pipe = pipeline(self.task, model=model_or_pipeline, tokenizer=tokenizer, **kwargs)
        elif isinstance(model_or_pipeline, Pipeline):
            pipe = pipeline(self.task, model=model_or_pipeline.model, tokenizer=model_or_pipeline.tokenizer, **kwargs)
        else:
            if model_or_pipeline is None:
                pipe = pipeline(self.task, **kwargs)
            else:
                pipe = model_or_pipeline
            if tokenizer is not None:
                logger.warning("Ignoring the value of the `tokenizer` argument.")
        if pipe.task != self.task:
            raise ValueError(
                f"Incompatible `model_or_pipeline`. Please specify `model_or_pipeline` compatible with the `{self.task}` task."
            )

        # Prepare metric.
        if metric is None:
            if self.default_metric_name is None:
                raise ValueError(
                    "`Evaluator` doesn't specify a default metric. Please specify a valid `metric` argument."
                )
            metric = load(self.default_metric_name)
        elif isinstance(metric, str):
            metric = load(metric)

        # Prepare reference.
        data = data.map(
            partial(self._tokenize_and_align_labels, tokenizer=pipe.tokenizer, input_column=input_column),
            batched=True,
            num_proc=None,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

        features = data.features

        # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
        # Otherwise, we have to get the list of labels manually.
        labels_are_int = isinstance(features[label_column].feature, ClassLabel)
        if labels_are_int:
            label_list = features[label_column].feature.names  # list of string labels
            ref_to_labels = {i: label for i, label in enumerate(label_list)}
        elif features[label_column].feature.dtype.startswith("int"):
            raise NotImplementedError(
                "References provided as int, but the reference column is not instanciated as a Sequence of ClassLabel."
            )
        else:
            # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
            # unique labels.
            # An example is labels as ["PER", "PER", "O", "LOC", "O", "LOC", "O"], e.g. in polyglot_ner dataset
            # `ref_to_labels` will map labels to labels, here {"LOC": "LOC", "PER": "PER", "O": "O"}
            # Normally `ref_to_labels` would just be e.g. {0: "O", 1: "LOC", 2: "PER"}
            unique_labels = set()
            for label in data[label_column]:
                unique_labels = unique_labels | set(label)
            ref_to_labels = {label: label for label in unique_labels}

        # Core computations.
        references = [[ref_to_labels[l] for l in label] for label in data[label_column]]
        predictions = self._compute_predictions(pipe, data[input_column], label_mapping=label_mapping, join_by=join_by)
        result = metric.compute(predictions=predictions, references=references)

        if strategy == "bootstrap":
            metric_keys = result.keys()
            bootstrap_dict = Evaluator._compute_confidence_interval(
                predictions,
                references,
                metric,
                metric_keys,
                confidence_level,
                n_resamples,
                random_state,
            )
            for key in metric_keys:
                bootstrap_dict[key]["score"] = result[key]

            return bootstrap_dict
        return result

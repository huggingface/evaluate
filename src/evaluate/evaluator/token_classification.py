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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from datasets import ClassLabel, Dataset, Sequence
from typing_extensions import Literal

from evaluate.evaluator.utils import DatasetColumn

from .base import Evaluator


class TokenClassificationEvaluator(Evaluator):
    """
    Token classification evaluator.

    This token classification evaluator can currently be loaded from [`evaluator`] using the default task name
    `token-classification`.

    Methods in this class assume a data format compatible with the [`TokenClassificationPipeline`].
    """

    PIPELINE_KWARGS = {"ignore_labels": []}

    def __init__(self, task="token-classification", default_metric_name=None):
        super().__init__(task, default_metric_name=default_metric_name)

    def predictions_processor(self, predictions: List[List[Dict]], words: List[List[str]], join_by: str):
        """
        Transform the pipeline predictions into a list of predicted labels of the same length as the true labels.

        Args:
            predictions (List[List[Dict]]): List of pipeline predictions, where each token has been labeled.
            words (List[List[str]]): Original input data to the pipeline, used to build predicted labels of the same length.
            join_by (str): String to use to join two words. In English, it will typically be " ".

        Returns:
            Dict: a dictionary holding the predictions
        """
        preds = []

        # iterate over the data rows
        for i, prediction in enumerate(predictions):
            pred_processed = []

            # get a list of tuples giving the indexes of the start and end character of each word
            words_offsets = self.words_to_offsets(words[i], join_by)

            token_index = 0
            for word_offset in words_offsets:
                # for each word, we may keep only the predicted label for the first token, discard the others
                while prediction[token_index]["start"] < word_offset[0]:
                    token_index += 1

                if prediction[token_index]["start"] > word_offset[0]:  # bad indexing
                    pred_processed.append("O")
                elif prediction[token_index]["start"] == word_offset[0]:
                    pred_processed.append(prediction[token_index]["entity"])

            preds.append(pred_processed)

        return {"predictions": preds}

    def words_to_offsets(self, words: List[str], join_by: str):
        """
        Convert a list of words to a list of offsets, where word are joined by `join_by`.

        Args:
            words (List[str]): List of words to get offsets from.
            join_by (str): String to insert between words.

        Returns:
            List[Tuple[int, int]]: List of the characters (start index, end index) for each of the words.
        """
        offsets = []

        start = 0
        for word in words:
            end = start + len(word) - 1
            offsets.append((start, end))
            start = end + len(join_by) + 1

        return offsets

    def prepare_data(self, data: Union[str, Dataset], input_column: str, label_column: str, join_by: str):
        super().prepare_data(data, input_column, label_column)

        if not isinstance(data.features[input_column], Sequence) or not isinstance(
            data.features[label_column], Sequence
        ):
            raise ValueError(
                "TokenClassificationEvaluator expects the input and label columns to be provided as lists."
            )

        # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
        # Otherwise, we have to get the list of labels manually.
        labels_are_int = isinstance(data.features[label_column].feature, ClassLabel)
        if labels_are_int:
            label_list = data.features[label_column].feature.names  # list of string labels
            id_to_label = {i: label for i, label in enumerate(label_list)}
            references = [[id_to_label[label_id] for label_id in label_ids] for label_ids in data[label_column]]
        elif data.features[label_column].feature.dtype.startswith("int"):
            raise NotImplementedError(
                "References provided as integers, but the reference column is not a Sequence of ClassLabels."
            )
        else:
            # In the event the labels are not a `Sequence[ClassLabel]`, we have already labels as strings
            # An example is labels as ["PER", "PER", "O", "LOC", "O", "LOC", "O"], e.g. in polyglot_ner dataset
            references = data[label_column]

        metric_inputs = {"references": references}
        data = data.map(lambda x: {input_column: join_by.join(x[input_column])})
        pipeline_inputs = DatasetColumn(data, input_column)

        return metric_inputs, pipeline_inputs

    def prepare_pipeline(
        self,
        model_or_pipeline: Union[str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"],  # noqa: F821
        tokenizer: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"] = None,  # noqa: F821
        feature_extractor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"] = None,  # noqa: F821
        device: int = None,
    ):
        pipe = super().prepare_pipeline(model_or_pipeline, tokenizer, feature_extractor, device)

        # check the pipeline outputs start characters in its predictions
        dummy_output = pipe(["2003 New York Gregory"], **self.PIPELINE_KWARGS)
        if dummy_output[0][0]["start"] is None:
            raise ValueError(
                "TokenClassificationEvaluator supports only pipelines giving 'start' index as a pipeline output (got None). "
                "Transformers pipelines with a slow tokenizer will raise this error."
            )

        return pipe

    def compute(
        self,
        model_or_pipeline: Union[
            str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"  # noqa: F821
        ] = None,
        data: Union[str, Dataset] = None,
        data_split: str = None,
        metric: Union[str, "EvaluationModule"] = None,  # noqa: F821
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,  # noqa: F821
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        device: Optional[int] = None,
        random_state: Optional[int] = None,
        input_column: str = "tokens",
        label_column: str = "ner_tags",
        join_by: Optional[str] = " ",
    ) -> Tuple[Dict[str, float], Any]:
        """
        Compute the metric for a given pipeline and dataset combination.

        Args:
            model_or_pipeline (`str` or `Pipeline` or `Callable` or `PreTrainedModel` or `TFPreTrainedModel`, defaults to `None`):
                If the argument in not specified, we initialize the default pipeline for the task (in this case
                `token-classification`). If the argument is of the type `str` or
                is a model instance, we use it to initialize a new `Pipeline` with the given model. Otherwise we assume the
                argument specifies a pre-initialized pipeline.
            data (`str` or `Dataset`, defaults to `None`):
                Specifies the dataset we will run evaluation on. If it is of type `str`, we treat it as the dataset
                name, and load it. Otherwise we assume it represents a pre-loaded dataset.
            metric (`str` or `EvaluationModule`, defaults to `None`):
                Specifies the metric we use in evaluator. If it is of type `str`, we treat it as the metric name, and
                load it. Otherwise we assume it represents a pre-loaded metric.
            tokenizer (`str` or `PreTrainedTokenizer`, *optional*, defaults to `None`):
                Argument can be used to overwrite a default tokenizer if `model_or_pipeline` represents a model for
                which we build a pipeline. If `model_or_pipeline` is `None` or a pre-initialized pipeline, we ignore
                this argument.
            strategy (`Literal["simple", "bootstrap"]`, defaults to "simple"):
                specifies the evaluation strategy. Possible values are:

                - `"simple"` - we evaluate the metric and return the scores.
                - `"bootstrap"` - on top of computing the metric scores, we calculate the confidence interval for each
                of the returned metric keys, using `scipy`'s `bootstrap` method
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html.
            confidence_level (`float`, defaults to `0.95`):
                The `confidence_level` value passed to `bootstrap` if `"bootstrap"` strategy is chosen.
            n_resamples (`int`, defaults to `9999`):
                The `n_resamples` value passed to `bootstrap` if `"bootstrap"` strategy is chosen.
            device (`int`, defaults to `None`):
                Device ordinal for CPU/GPU support of the pipeline. Setting this to -1 will leverage CPU, a positive
                integer will run the model on the associated CUDA device ID. If`None` is provided it will be inferred and
                CUDA:0 used if available, CPU otherwise.
            random_state (`int`, *optional*, defaults to `None`):
                The `random_state` value passed to `bootstrap` if `"bootstrap"` strategy is chosen. Useful for
                debugging.
            input_column (`str`, defaults to `"tokens"`):
                the name of the column containing the tokens feature in the dataset specified by `data`.
            label_column (`str`, defaults to `"label"`):
                the name of the column containing the labels in the dataset specified by `data`.
            join_by (`str`, *optional*, defaults to `" "`):
                This evaluator supports dataset whose input column is a list of words. This parameter specifies how to join
                words to generate a string input. This is especially useful for languages that do not separate words by a space.

        Return:
            A `Dict`. The keys represent metric keys calculated for the `metric` spefied in function arguments. For the
            `"simple"` strategy, the value is the metric score. For the `"bootstrap"` strategy, the value is a `Dict`
            containing the score, the confidence interval and the standard error calculated for each metric key.

        The dataset input and label columns are expected to be formatted as a list of words and a list of labels respectively, following [conll2003 dataset](https://huggingface.co/datasets/conll2003). Datasets whose inputs are single strings, and labels are a list of offset are not supported.

        Examples:
        ```python
        >>> from evaluate import evaluator
        >>> from datasets import load_dataset
        >>> task_evaluator = evaluator("token-classification")
        >>> data = load_dataset("conll2003", split="validation[:2]")
        >>> results = task_evaluator.compute(
        >>>     model_or_pipeline="elastic/distilbert-base-uncased-finetuned-conll03-english",
        >>>     data=data,
        >>>     metric="seqeval",
        >>> )
        ```

        <Tip>

        For example, the following dataset format is accepted by the evaluator:

        ```python
        dataset = Dataset.from_dict(
            mapping={
                "tokens": [["New", "York", "is", "a", "city", "and", "Felix", "a", "person", "."]],
                "ner_tags": [[1, 2, 0, 0, 0, 0, 3, 0, 0, 0]],
            },
            features=Features({
                "tokens": Sequence(feature=Value(dtype="string")),
                "ner_tags": Sequence(feature=ClassLabel(names=["O", "B-LOC", "I-LOC", "B-PER", "I-PER"])),
                }),
        )
        ```

        </Tip>

        <Tip warning={true}>

        For example, the following dataset format is **not** accepted by the evaluator:

        ```python
        dataset = Dataset.from_dict(
            mapping={
                "tokens": [["New York is a city and Felix a person."]],
                "starts": [[0, 23]],
                "ends": [[7, 27]],
                "ner_tags": [["LOC", "PER"]],
            },
            features=Features({
                "tokens": Value(dtype="string"),
                "starts": Sequence(feature=Value(dtype="int32")),
                "ends": Sequence(feature=Value(dtype="int32")),
                "ner_tags": Sequence(feature=Value(dtype="string")),
            }),
        )
        ```

        </Tip>
        """
        result = {}

        # Prepare inputs
        data = self.load_data(data=data, data_split=data_split)
        metric_inputs, pipe_inputs = self.prepare_data(
            data=data, input_column=input_column, label_column=label_column, join_by=join_by
        )
        pipe = self.prepare_pipeline(model_or_pipeline=model_or_pipeline, tokenizer=tokenizer, device=device)
        metric = self.prepare_metric(metric)

        # Compute predictions
        predictions, perf_results = self.call_pipeline(pipe, pipe_inputs)
        predictions = self.predictions_processor(predictions, data[input_column], join_by)

        metric_inputs.update(predictions)

        # Compute metrics from references and predictions
        metric_results = self.compute_metric(
            metric=metric,
            metric_inputs=metric_inputs,
            strategy=strategy,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            random_state=random_state,
        )

        result.update(metric_results)
        result.update(perf_results)

        return result

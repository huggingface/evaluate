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

from typing import Dict, Tuple

from datasets import Dataset

from .base import Evaluator
from .utils import DatasetColumn


TASK_DOCUMENTATION_KWARGS = r"""
        input_column (`str`, defaults to `"text"`):
            the name of the column containing the input text in the dataset specified by `data`.
        generation_kwargs (`Dict`, *optional*, defaults to `None`):
            The generation kwargs are passed to the pipeline and set the text generation strategy.
"""


class TextGenerationEvaluator(Evaluator):
    """
    Text generation evaluator.
    This Text generation evaluator can currently be loaded from [`evaluator`] using the default task name
    `text-generation`.
    Methods in this class assume a data format compatible with the [`~transformers.TextGenerationPipeline`].
    """

    def predictions_processor(self, predictions, *args, **kwargs):
        """
        Args:
            predictions: A list of lists of dicts

        Returns:
            `dict`: All the generated texts are flattened and stored under the "data" key.
        """
        return {"data": [pred[f"{self.predictions_prefix}_text"] for pred_list in predictions for pred in pred_list]}

    def __init__(self, task="text-generation", default_metric_name=None, predictions_prefix: str = "generated"):
        super().__init__(task=task, default_metric_name=default_metric_name)
        self.predictions_prefix = predictions_prefix

    def prepare_data(self, data: Dataset, input_column: str, *args, **kwargs) -> Tuple[Dict, DatasetColumn]:
        """
        Prepare data.

        Args:
            data ([`Dataset`]):
                Specifies the dataset we will run evaluation on.
            input_column (`str`, defaults to `"text"`):
                The name of the column containing the text feature in the dataset specified by `data`.
        Returns:
            `dict`:  metric inputs.
            `list`:  pipeline inputs.
        """

        self.check_required_columns(data, {"input_column": input_column})

        return {}, DatasetColumn(data, input_column)

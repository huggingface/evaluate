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

from typing import List, Optional
import evaluate
import numpy as np

_DESCRIPTION = """
This metric computes the mean Intersection-Over-Union (mIoU) for image segmentation tasks.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (List[int]): Predicted class indices.
    references (List[int]): Ground truth class indices.
    num_labels (int): Total number of classes.

Returns:
    mean_iou (float): The mean IoU across all classes.
"""

class MeanIoU(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation="",
            inputs_description=_KWARGS_DESCRIPTION,
            features=evaluate.Features({
                'predictions': evaluate.Sequence(evaluate.Value('int64')),
                'references': evaluate.Sequence(evaluate.Value('int64')),
            }),
            reference_urls=[]
        )

    def _compute(self, predictions, references, num_labels):
        predictions = np.array(predictions)
        references = np.array(references)
        iou_list = []

        for label in range(num_labels):
            tp = np.sum((predictions == label) & (references == label))
            fp = np.sum((predictions == label) & (references != label))
            fn = np.sum((predictions != label) & (references == label))

            denom = tp + fp + fn + 1e-10  # Prevent division by zero
            iou = tp / denom if denom != 0 else 0.0
            iou_list.append(iou)

        mean_iou = np.mean(iou_list)
        return {"mean_iou": mean_iou}

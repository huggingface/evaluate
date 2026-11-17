# Copyright 2026 The HuggingFace Evaluate Authors.
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

import numpy as np
import pytest

import evaluate
from metrics.smape.smape import symmetric_mean_absolute_percentage_error


@pytest.mark.parametrize(
    "references,predictions,expected",
    [
        ([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0], 0.5787878787878785),
        ([-1.0, 0.0, 2.0], [1.0, 0.0, -2.0], 4 / 3),
        ([0.0, 1.0], [0.0, 1.0], 0.0),
    ],
)
def test_smape_scalar(references, predictions, expected):
    metric = evaluate.load("./metrics/smape")
    result = metric.compute(references=references, predictions=predictions)
    assert result["smape"] == pytest.approx(expected)


@pytest.mark.parametrize("sample_weight,first_output", [(None, 0.5), ([1.0, 3.0], 0.75)])
@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average", [3.0, 1.0]])
def test_smape_multioutput(sample_weight, first_output, multioutput):
    metric = evaluate.load("./metrics/smape", "multilist")
    result = metric.compute(
        references=[[1.0, 2.0], [3.0, 4.0]],
        predictions=[[1.0, 4.0], [1.0, 2.0]],
        sample_weight=sample_weight,
        multioutput=multioutput,
    )
    expected = [first_output, 2 / 3]
    if isinstance(multioutput, str):
        if multioutput == "uniform_average":
            expected = sum(expected) / 2
    else:
        expected = (3 * expected[0] + expected[1]) / 4
    np.testing.assert_allclose(result["smape"], expected)


def test_smape_accepts_single_output_column():
    result = symmetric_mean_absolute_percentage_error([1.0, 3.0], [[1.0], [1.0]])
    assert result == pytest.approx(0.5)


@pytest.mark.parametrize(
    "references,predictions,kwargs",
    [
        ([1.0, 2.0], [1.0], {}),
        ([[1.0, 2.0]], [[1.0]], {}),
        ([1.0, 2.0], [1.0, 2.0], {"sample_weight": [1.0]}),
        ([[1.0, 2.0]], [[1.0, 2.0]], {"multioutput": [1.0]}),
        ([1.0, 2.0], [1.0, 2.0], {"multioutput": "invalid"}),
        ([1.0, np.nan], [1.0, 2.0], {}),
    ],
)
def test_smape_rejects_invalid_inputs(references, predictions, kwargs):
    with pytest.raises(ValueError):
        symmetric_mean_absolute_percentage_error(references, predictions, **kwargs)

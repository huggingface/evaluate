# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
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

# Lint as: python3
""" DatasetInfo and MetricInfo record information we know about a dataset and a metric.

This includes things that we know about the dataset statically, i.e.:
 - description
 - canonical location
 - does it have validation and tests splits
 - size
 - etc.

This also includes the things that can and should be computed once we've
processed the dataset as well:
 - number of examples (in each split)
 - etc.
"""

import dataclasses
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from datasets.features import Features, Value

from . import config
from .utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class SupervisedKeysData:
    input: str = ""
    output: str = ""


@dataclass
class DownloadChecksumsEntryData:
    key: str = ""
    value: str = ""


class MissingCachedSizesConfigError(Exception):
    """The expected cached sizes of the download file are missing."""


class NonMatchingCachedSizesError(Exception):
    """The prepared split doesn't have expected sizes."""


@dataclass
class MetricInfo:
    """Information about a metric.

    `MetricInfo` documents a metric, including its name, version, and features.
    See the constructor arguments and properties for a full list.

    Note: Not all fields are known on construction and may be updated later.
    """

    # Set in the dataset scripts
    description: str
    citation: str
    features: Features
    inputs_description: str = field(default_factory=str)
    homepage: str = field(default_factory=str)
    license: str = field(default_factory=str)
    codebase_urls: List[str] = field(default_factory=list)
    reference_urls: List[str] = field(default_factory=list)
    streamable: bool = False
    format: Optional[str] = None

    # Set later by the builder
    metric_name: Optional[str] = None
    config_name: Optional[str] = None
    experiment_id: Optional[str] = None

    def __post_init__(self):
        if self.format is not None:
            for key, value in self.features.items():
                if not isinstance(value, Value):
                    raise ValueError(
                        f"When using 'numpy' format, all features should be a `datasets.Value` feature. "
                        f"Here {key} is an instance of {value.__class__.__name__}"
                    )

    def write_to_directory(self, metric_info_dir):
        """Write `MetricInfo` as JSON to `metric_info_dir`.
        Also save the license separately in LICENCE.
        """
        with open(os.path.join(metric_info_dir, config.METRIC_INFO_FILENAME), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f)

        with open(os.path.join(metric_info_dir, config.LICENSE_FILENAME), "w", encoding="utf-8") as f:
            f.write(self.license)

    @classmethod
    def from_directory(cls, metric_info_dir) -> "MetricInfo":
        """Create MetricInfo from the JSON file in `metric_info_dir`.

        Args:
            metric_info_dir: `str` The directory containing the metadata file. This
                should be the root directory of a specific dataset version.
        """
        logger.info(f"Loading Metric info from {metric_info_dir}")
        if not metric_info_dir:
            raise ValueError("Calling MetricInfo.from_directory() with undefined metric_info_dir.")

        with open(os.path.join(metric_info_dir, config.METRIC_INFO_FILENAME), encoding="utf-8") as f:
            metric_info_dict = json.load(f)
        return cls.from_dict(metric_info_dict)

    @classmethod
    def from_dict(cls, metric_info_dict: dict) -> "MetricInfo":
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in metric_info_dict.items() if k in field_names})

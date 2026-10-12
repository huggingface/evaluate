# Copyright 2020 The HuggingFace Evaluate Authors.
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

"""Tests for the cookiecutter template used by `evaluate-cli create`.

Regression test for issue #514: with the default empty ``dataset_name``, the
template used to render ``datasets:\\n- `` (a list containing None), which the
Hub rejects with ``"datasets[0]" must be a string``.
"""

import pathlib

import pytest
import yaml
from cookiecutter.main import cookiecutter

BASE_CONTEXT = {
    "module_name": "My Metric",
    "module_type": "module",
    "module_slug": "my_metric",
    "module_class_name": "MyMetric",
    "namespace": "",
}


def _render_readme(dataset_name: str, tmp_path) -> str:
    out = cookiecutter(
        str(pathlib.Path(__file__).parent.parent / "templates"),
        no_input=True,
        extra_context={**BASE_CONTEXT, "dataset_name": dataset_name},
        output_dir=str(tmp_path),
    )
    return next(pathlib.Path(out).rglob("README.md")).read_text()


def _frontmatter(readme: str) -> dict:
    parts = readme.split("---")
    assert len(parts) >= 3, "README must start with a YAML frontmatter block"
    return yaml.safe_load(parts[1])


def test_template_with_empty_dataset_name_omits_datasets(tmp_path):
    readme = _render_readme("", tmp_path)
    fm = _frontmatter(readme)
    datasets = fm.get("datasets", None)
    assert datasets is None or all(isinstance(d, str) and d for d in datasets), (
        f"'datasets' must be absent or a list of non-empty strings, got {datasets!r}"
    )


def test_template_with_dataset_name_keeps_datasets(tmp_path):
    readme = _render_readme("squad", tmp_path)
    fm = _frontmatter(readme)
    assert fm["datasets"] == ["squad"]

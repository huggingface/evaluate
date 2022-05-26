from unittest import TestCase
from unittest.mock import patch

import pytest

from evaluate.hub import push_to_hub
from tests.test_metric import DummyMetric


minimum_metadata = {
    "model-index": [
        {
            "results": [
                {
                    "task": {"type": "task_type"},
                    "dataset": {"type": "dataset_type", "name": "dataset_name"},
                    "metrics": [
                        {"type": "metric", "value": 1.0},
                    ],
                }
            ]
        }
    ]
}

extras_metadata = {
    "model-index": [
        {
            "results": [
                {
                    "task": {"type": "task_type", "name": "task_name"},
                    "dataset": {
                        "type": "dataset_type",
                        "name": "dataset_name",
                        "config": "fr",
                        "split": "test",
                        "revision": "abc",
                        "args": {"a": 1, "b": 2},
                    },
                    "metrics": [
                        {
                            "type": "metric",
                            "value": 1.0,
                            "name": "dummy_metric",
                            "config": "default",
                            "args": {"hello": 1, "world": 2},
                        },
                    ],
                }
            ]
        }
    ]
}


@patch("evaluate.hub.metadata_update")
class TestHub(TestCase):
    def setUp(self):
        self.metric = DummyMetric()
        self.metric.add()
        self.args = {"hello": 1, "world": 2}
        self.result = self.metric.compute(**self.args)

    def test_push_metric_required_arguments(self, metadata_update):
        push_to_hub(
            to="username/repo",
            metric_value=self.result["accuracy"],
            metric_type=self.metric.type,
            dataset_name="dataset_name",
            dataset_type="dataset_type",
            task_type="task_type",
        )

        metadata_update.assert_called_once_with(repo_id="username/repo", metadata=minimum_metadata, overwrite=True)

    def test_push_metric_missing_arguments(self, metadata_update):
        with pytest.raises(TypeError):
            push_to_hub(
                to="username/repo",
                metric_value=self.result["accuracy"],
                metric_type=self.metric.type,
                dataset_name="dataset_name",
                dataset_type="dataset_type",
            )

    def test_push_metric_invalid_arguments(self, metadata_update):
        with pytest.raises(TypeError):
            push_to_hub(
                to="username/repo",
                metric_value=self.result["accuracy"],
                metric_type=self.metric.type,
                dataset_name="dataset_name",
                dataset_type="dataset_type",
                task_type="task_type",
                random_value="incorrect",
            )

    def test_push_metric_extra_arguments(self, metadata_update):
        push_to_hub(
            to="username/repo",
            metric_value=self.result["accuracy"],
            metric_type=self.metric.type,
            dataset_name="dataset_name",
            dataset_type="dataset_type",
            dataset_config="fr",
            dataset_split="test",
            dataset_revision="abc",
            dataset_args={"a": 1, "b": 2},
            task_type="task_type",
            task_name="task_name",
            metric_name=self.metric.name,
            metric_config=self.metric.config_name,
            metric_args=self.args,
        )

        metadata_update.assert_called_once_with(repo_id="username/repo", metadata=extras_metadata, overwrite=True)

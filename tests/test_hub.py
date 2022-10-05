from unittest import TestCase
from unittest.mock import patch

import pytest
import requests

from evaluate.hub import push_to_hub
from tests.test_metric import DummyMetric


minimum_metadata = {
    "model-index": [
        {
            "results": [
                {
                    "task": {"type": "dummy-task"},
                    "dataset": {"type": "dataset_type", "name": "dataset_name"},
                    "metrics": [
                        {"type": "dummy_metric", "value": 1.0, "name": "Pretty Metric Name"},
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
                    "task": {"type": "dummy-task", "name": "task_name"},
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
                            "type": "dummy_metric",
                            "value": 1.0,
                            "name": "Pretty Metric Name",
                            "config": "default",
                            "args": {"hello": 1, "world": 2},
                        },
                    ],
                }
            ]
        }
    ]
}


@patch("evaluate.hub.HF_HUB_ALLOWED_TASKS", ["dummy-task"])
@patch("evaluate.hub.dataset_info", lambda x: True)
@patch("evaluate.hub.model_info", lambda x: True)
@patch("evaluate.hub.metadata_update")
class TestHub(TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def setUp(self):
        self.metric = DummyMetric()
        self.metric.add()
        self.args = {"hello": 1, "world": 2}
        self.result = self.metric.compute()

    def test_push_metric_required_arguments(self, metadata_update):
        push_to_hub(
            model_id="username/repo",
            metric_value=self.result["accuracy"],
            metric_name="Pretty Metric Name",
            metric_type=self.metric.name,
            dataset_name="dataset_name",
            dataset_type="dataset_type",
            task_type="dummy-task",
        )

        metadata_update.assert_called_once_with(repo_id="username/repo", metadata=minimum_metadata, overwrite=False)

    def test_push_metric_missing_arguments(self, metadata_update):
        with pytest.raises(TypeError):
            push_to_hub(
                model_id="username/repo",
                metric_value=self.result["accuracy"],
                metric_name="Pretty Metric Name",
                metric_type=self.metric.name,
                dataset_name="dataset_name",
                dataset_type="dummy-task",
            )

    def test_push_metric_invalid_arguments(self, metadata_update):
        with pytest.raises(TypeError):
            push_to_hub(
                model_id="username/repo",
                metric_value=self.result["accuracy"],
                metric_name="Pretty Metric Name",
                metric_type=self.metric.name,
                dataset_name="dataset_name",
                dataset_type="dataset_type",
                task_type="dummy-task",
                random_value="incorrect",
            )

    def test_push_metric_extra_arguments(self, metadata_update):
        push_to_hub(
            model_id="username/repo",
            metric_value=self.result["accuracy"],
            metric_name="Pretty Metric Name",
            metric_type=self.metric.name,
            dataset_name="dataset_name",
            dataset_type="dataset_type",
            dataset_config="fr",
            dataset_split="test",
            dataset_revision="abc",
            dataset_args={"a": 1, "b": 2},
            task_type="dummy-task",
            task_name="task_name",
            metric_config=self.metric.config_name,
            metric_args=self.args,
        )

        metadata_update.assert_called_once_with(repo_id="username/repo", metadata=extras_metadata, overwrite=False)

    def test_push_metric_invalid_task_type(self, metadata_update):
        with pytest.raises(ValueError):
            push_to_hub(
                model_id="username/repo",
                metric_value=self.result["accuracy"],
                metric_name="Pretty Metric Name",
                metric_type=self.metric.name,
                dataset_name="dataset_name",
                dataset_type="dataset_type",
                task_type="audio-classification",
            )

    def test_push_metric_invalid_dataset_type(self, metadata_update):
        with patch("evaluate.hub.dataset_info") as mock_dataset_info:
            mock_dataset_info.side_effect = requests.HTTPError()
            push_to_hub(
                model_id="username/repo",
                metric_value=self.result["accuracy"],
                metric_name="Pretty Metric Name",
                metric_type=self.metric.name,
                dataset_name="dataset_name",
                dataset_type="dataset_type",
                task_type="dummy-task",
            )

            assert "Dataset dataset_type not found on the Hub at hf.co/datasets/dataset_type" in self._caplog.text
            metadata_update.assert_called_once_with(
                repo_id="username/repo", metadata=minimum_metadata, overwrite=False
            )

    def test_push_metric_invalid_model_id(self, metadata_update):
        with patch("evaluate.hub.model_info") as mock_model_info:
            mock_model_info.side_effect = requests.HTTPError()
            with pytest.raises(ValueError):
                push_to_hub(
                    model_id="username/bad-repo",
                    metric_value=self.result["accuracy"],
                    metric_name="Pretty Metric Name",
                    metric_type=self.metric.name,
                    dataset_name="dataset_name",
                    dataset_type="dataset_type",
                    task_type="dummy-task",
                )

from unittest import TestCase
from unittest.mock import patch

import pytest

from evaluate.hub import push_to_hub, get_allowed_tasks
from tests.test_metric import DummyMetric


minimum_metadata = {
    "model-index": [
        {
            "results": [
                {
                    "task": {"type": "dummy-task"},
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


@patch("evaluate.hub.get_allowed_tasks", lambda x: ['dummy-task'])
@patch("evaluate.hub.metadata_update")
@patch("evaluate.hub.known_task_ids")
class TestHub(TestCase):
    def setUp(self):
        self.metric = DummyMetric()
        self.metric.add()
        self.args = {"hello": 1, "world": 2}
        self.result = self.metric.compute()

    def test_push_metric_required_arguments(self, known_task_ids, metadata_update):
        push_to_hub(
            repo_id="username/repo",
            metric_value=self.result["accuracy"],
            metric_type=self.metric.type,
            dataset_name="dataset_name",
            dataset_type="dataset_type",
            task_type="dummy-task",
        )

        metadata_update.assert_called_once_with(repo_id="username/repo", metadata=minimum_metadata, overwrite=False)

    def test_push_metric_missing_arguments(self, known_task_ids, metadata_update):
        with pytest.raises(TypeError):
            push_to_hub(
                repo_id="username/repo",
                metric_value=self.result["accuracy"],
                metric_type=self.metric.type,
                dataset_name="dataset_name",
                dataset_type="dummy-task",
            )

    def test_push_metric_invalid_arguments(self, known_task_ids, metadata_update):
        with pytest.raises(TypeError):
            push_to_hub(
                repo_id="username/repo",
                metric_value=self.result["accuracy"],
                metric_type=self.metric.type,
                dataset_name="dataset_name",
                dataset_type="dataset_type",
                task_type="dummy-task",
                random_value="incorrect",
            )

    def test_push_metric_extra_arguments(self, known_task_ids, metadata_update):
        push_to_hub(
            repo_id="username/repo",
            metric_value=self.result["accuracy"],
            metric_type=self.metric.type,
            dataset_name="dataset_name",
            dataset_type="dataset_type",
            dataset_config="fr",
            dataset_split="test",
            dataset_revision="abc",
            dataset_args={"a": 1, "b": 2},
            task_type="dummy-task",
            task_name="task_name",
            metric_name=self.metric.name,
            metric_config=self.metric.config_name,
            metric_args=self.args,
        )

        metadata_update.assert_called_once_with(repo_id="username/repo", metadata=extras_metadata, overwrite=False)

    def test_push_metric_invalid_task_type(self, known_task_ids, metadata_update):
        with pytest.raises(ValueError):
            push_to_hub(
                repo_id="username/repo",
                metric_value=self.result["accuracy"],
                metric_type=self.metric.type,
                dataset_name="dataset_name",
                dataset_type="dataset_type",
                task_type="audio-classification",
            )


@pytest.mark.parametrize("tasks_dict, expected", [
    ({'a': {'subtasks': ['b', 'c']}}, ['a', 'b', 'c']),
    ({'a': {}, 'b': {'subtasks': ['c', 'd']}, 'e': {}}, ['a', 'b', 'e', 'c', 'd'])
])
def test_get_allowed_tasks(tasks_dict, expected):
    tasks = get_allowed_tasks(tasks_dict)

    assert tasks == expected

import evaluate
from evaluate.evaluation_suite import SubTask


class Suite(evaluate.EvaluationSuite):

    def __init__(self, name):
        super().__init__(name)

        self.preprocessor = lambda x: {"text": x["text"].lower()}
        self.suite = [
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="sst2",
                split="validation[:2]",
                args_for_task={
                    "metric": "glue",
                    "input_column": "sentence",
                    "label_column": "label",
                    "config_name": "sst2",
                    "label_mapping": {
                        "NEGATIVE": 0.0,
                        "POSITIVE": 1.0
                    }
                }
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="qqp",
                split="validation[:2]",
                args_for_task={
                    "metric": "glue",
                    "input_column": "question1",
                    "second_input_column": "question2",
                    "label_column": "label",
                    "config_name": "qqp",
                    "label_mapping": {
                        "NEGATIVE": 0.0,
                        "POSITIVE": 1.0
                    }
                }
            )
        ]

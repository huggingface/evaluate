from evaluate.evaluation_suite import SubTask
from evaluate.visualization import radar_plot

from intel_evaluate_extension.evaluation_suite.model_card_suite import ModelCardSuiteResults

_HEADER =  "GLUE/AdvGlue Evaluation Results"

_DESCRIPTION = """
The suite compares the GLUE results with Adversarial GLUE (AdvGLUE), a multi-task benchmark 
meaure the vulnerabilities of modern large-scale language models under various types of adversarial attacks."""


class Suite(ModelCardSuiteResults):
    def __init__(self, name):
        super().__init__(name)
        self.result_keys = ["accuracy", "f1"]
        self.preprocessor = lambda x: {"text": x["text"].lower()}
        self.suite = [
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="sst2",
                split="validation[:5]",
                args_for_task={
                    "metric": "glue",
                    "input_column": "sentence",
                    "label_column": "label",
                    "config_name": "sst2",
                    "label_mapping": {
                        "LABEL_0": 0.0,
                        "LABEL_1": 1.0
                    }
                }
            ),
            SubTask(
                task_type="text-classification",
                data="adv_glue",
                subset="adv_sst2",
                split="validation[:5]",
                args_for_task={
                    "metric": "glue",
                    "input_column": "sentence",
                    "label_column": "label",
                    "config_name": "sst2",
                    "label_mapping": {
                        "LABEL_0": 0.0,
                        "LABEL_1": 1.0
                    }
                }
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="qqp",
                split="validation[:5]",

                args_for_task={
                    "metric": "glue",
                    "input_column": "question1",
                    "second_input_column": "question2",
                    "label_column": "label",
                    "config_name": "qqp",
                    "label_mapping": {
                        "LABEL_0": 0,
                        "LABEL_1": 1
                    }
                }
            ),
            SubTask(
                task_type="text-classification",
                data="adv_glue",
                subset="adv_qqp",
                split="validation[:5]",
                args_for_task={
                    "metric": "glue",
                    "input_column": "question1",
                    "second_input_column": "question2",
                    "label_column": "label",
                    "config_name": "qqp",
                    "label_mapping": {
                        "LABEL_0": 0,
                        "LABEL_1": 1
                    }
                }
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="qnli",
                split="validation[:5]",
                args_for_task={
                    "metric": "glue",
                    "input_column": "question",
                    "second_input_column": "sentence",
                    "label_column": "label",
                    "config_name": "qnli",
                    "label_mapping": {
                        "LABEL_0": 0,
                        "LABEL_1": 1
                    }
                }
            ),
            SubTask(
                task_type="text-classification",
                data="adv_glue",
                subset="adv_qnli",
                split="validation[:5]",
                args_for_task={
                    "metric": "glue",
                    "input_column": "question",
                    "second_input_column": "sentence",
                    "label_column": "label",
                    "config_name": "qnli",
                    "label_mapping": {
                        "LABEL_0": 0,
                        "LABEL_1": 1
                    }
                }
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="rte",
                split="validation[:5]",
                args_for_task={
                    "metric": "glue",
                    "input_column": "sentence1",
                    "second_input_column": "sentence2",
                    "label_column": "label",
                    "config_name": "rte",
                    "label_mapping": {
                        "LABEL_0": 0,
                        "LABEL_1": 1
                    }
                }
            ),
            SubTask(
                task_type="text-classification",
                data="adv_glue",
                subset="adv_rte",
                split="validation[:5]",
                args_for_task={
                    "metric": "glue",
                    "input_column": "sentence1",
                    "second_input_column": "sentence2",
                    "label_column": "label",
                    "config_name": "rte",
                    "label_mapping": {
                        "LABEL_0": 0,
                        "LABEL_1": 1
                    }
                }
            ),
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="mnli",
                split="validation_mismatched[:5]",
                args_for_task={
                    "metric": "glue",
                    "input_column": "premise",
                    "second_input_column": "hypothesis",
                    "config_name": "mnli",
                    "label_mapping": {
                        "LABEL_0": 0,
                        "LABEL_1": 1,
                        "LABEL_2": 2
                    }
                }
            ),
            SubTask(
                task_type="text-classification",
                data="adv_glue",
                subset="adv_mnli",
                split="validation[:5]",
                args_for_task={
                    "metric": "glue",
                    "input_column": "premise",
                    "second_input_column": "hypothesis",
                    "config_name": "mnli",
                    "label_mapping": {
                        "LABEL_0": 0,
                        "LABEL_1": 1,
                        "LABEL_2": 2
                    }
                }
            ),
        ]

    def process_results(self, results):
        radar_data = [
            {"accuracy " + result["task_name"].split("/")[-1]: 
             result["accuracy"] for result in results[::2]},
            {"accuracy " + result["task_name"].replace("adv_", "").split("/")[-1]: 
             result["accuracy"] for result in results[1::2]}]
        return radar_plot(radar_data, ['GLUE', 'AdvGLUE'])

    def plot_results(self, results, model_or_pipeline):
        radar_data = self.process_results(results)
        graphic = radar_plot(radar_data, ['GLUE ' + model_or_pipeline,  'AdvGLUE ' + model_or_pipeline])
        return graphic

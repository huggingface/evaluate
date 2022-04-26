import json
import os
import re
import sys
from pathlib import Path

from datasets import Value

from .logging import get_logger


logger = get_logger(__name__)

REGEX_YAML_BLOCK = re.compile(r"---[\n\r]+([\S\s]*?)[\n\r]+---[\n\r]")


def infer_gradio_input_types(feature_types):
    """
    Maps metric feature types to input types for gradio Dataframes:
        - float/int -> numbers
        - string -> strings
        - any other -> json
    Note that json is not a native gradio type but will be treated as string that
    is then parsed as a json.
    """
    input_types = []
    for feature_type in feature_types:
        input_type = "json"
        if isinstance(feature_type, Value):
            if feature_type.dtype.startswith("int") or feature_type.dtype.startswith("float"):
                input_type = "number"
            elif feature_type.dtype == "string":
                input_type = "str"
        input_types.append(input_type)
    return input_types


def json_to_string_type(input_types):
    """Maps json input type to str."""
    return ["str" if i == "json" else i for i in input_types]


def parse_readme(filepath):
    """Parses a repositories README and removes"""
    if not os.path.exists(filepath):
        return "No README.md found."
    with open(filepath, "r") as f:
        text = f.read()
        match = REGEX_YAML_BLOCK.search(text)
        if match:
            text = text[match.end() :]
    return text


def parse_gradio_data(data, input_types):
    """Parses data from gradio Dataframe for use in metric."""
    metric_inputs = {}
    data.dropna(inplace=True)
    for feature_name, input_type in zip(data, input_types):
        if input_type == "json":
            metric_inputs[feature_name] = [json.loads(d) for d in data[feature_name].to_list()]
        elif input_type == "str":
            metric_inputs[feature_name] = [d.strip('"') for d in data[feature_name].to_list()]
        else:
            metric_inputs[feature_name] = data[feature_name]
    return metric_inputs


def parse_test_cases(test_cases, feature_names, input_types):
    """
    Parses test cases to be used in gradio Dataframe. Note that an apostrophe is added
    to strings to follow the format in json.
    """
    if len(test_cases) == 0:
        return None
    examples = []
    for test_case in test_cases:
        parsed_cases = []
        for feat, input_type in zip(feature_names, input_types):
            if input_type == "json":
                parsed_cases.append([str(element) for element in test_case[feat]])
            elif input_type == "str":
                parsed_cases.append(['"' + element + '"' for element in test_case[feat]])
            else:
                parsed_cases.append(test_case[feat])
        examples.append([list(i) for i in zip(*parsed_cases)])
    return examples


def launch_gradio_widget(metric):
    """Launches `metric` widget with Gradio."""

    try:
        import gradio as gr
    except ImportError as error:
        logger.error("To create a metric widget with Gradio make sure gradio is installed.")
        raise error

    local_path = Path(sys.path[0])

    (feature_names, feature_types) = zip(*metric.features.items())
    gradio_input_types = infer_gradio_input_types(feature_types)

    def compute(data):
        return metric.compute(**parse_gradio_data(data, gradio_input_types))

    iface = gr.Interface(
        fn=compute,
        inputs=gr.inputs.Dataframe(
            headers=feature_names,
            col_width=len(feature_names),
            row_count=2,
            datatype=json_to_string_type(gradio_input_types),
        ),
        outputs=gr.outputs.Textbox(label=metric.name),
        description=metric.info.description,
        title=f"Metric: {metric.name}",
        article=parse_readme(local_path / "README.md"),
        # TODO: load test cases and use them to populate examples
        # examples=[parse_test_cases(test_cases, feature_names, gradio_input_types)]
    )

    iface.launch()

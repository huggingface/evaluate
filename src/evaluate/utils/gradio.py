import json
import os
import re

from datasets import Value


REGEX_YAML_BLOCK = re.compile(r"---[\n\r]+([\S\s]*?)[\n\r]+---[\n\r]")


def infer_gradio_input_types(feature_types):
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
    return ["str" if i == "json" else i for i in input_types]


def parse_readme(filepath):
    if not os.path.exists(filepath):
        return "No README.md found."
    with open(filepath, "r") as f:
        text = f.read()
        match = REGEX_YAML_BLOCK.search(text)
        if match:
            text = text[match.end() :]
    return text


def parse_gradio_data(data, input_types):
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
    if len(test_cases) == 0:
        return None
    example_dataframes = []
    for test_case in test_cases:
        parsed_cases = []
        for feat, input_type in zip(feature_names, input_types):
            if input_type == "json":
                parsed_cases.append([str(element) for element in test_case[feat]])
            elif input_type == "str":
                parsed_cases.append(['"' + element + '"' for element in test_case[feat]])
            else:
                parsed_cases.append(test_case[feat])
        example_dataframes.append([list(i) for i in zip(*parsed_cases)])
    return example_dataframes

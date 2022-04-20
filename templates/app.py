from new_metric_script import NewMetric
from evaluate.utils import infer_gradio_input_types, json_to_string_type, parse_readme, parse_gradio_data, parse_test_cases
import gradio as gr
import os
from pathlib import Path
from tests import test_cases

folder_path = Path(os.path.dirname(os.path.realpath(__file__)))

metric = NewMetric()
(feature_names, feature_types) = zip(*metric.features.items())
gradio_input_types = infer_gradio_input_types(feature_types)

def compute(data):
    return metric.compute(**parse_gradio_data(data, gradio_input_types))

iface = gr.Interface(
  fn=compute, 
  inputs=gr.inputs.Dataframe(headers=feature_names,
                             col_width=len(feature_names),
                             row_count=2,
                             datatype=json_to_string_type(gradio_input_types)
                             ),
  outputs=gr.outputs.Textbox(label=metric.name),
  description=metric.info.description,
  title=f"Metric: {metric.name}",
  article=parse_readme(folder_path/"README.md"),
  examples=[parse_test_cases(test_cases, feature_names, gradio_input_types)]
 )

iface.launch()

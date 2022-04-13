from new_metric_script import NewMetric

import gradio as gr

metric = NewMetric()
metric_features = list(metric.features.keys())

def compute(data):
    metric_input = dict([(feat, data[feat]) for feat in metric_features])
    return metric.compute(**metric_input)
    
iface = gr.Interface(
  fn=compute, 
  inputs=gr.inputs.Dataframe(headers=metric_features, col_width=len(metric_features), datatype="number"),
  outputs=gr.outputs.Textbox(label=metric.name),
  description=metric.info.description,
  article=metric.info.citation,
 )
 
iface.launch()

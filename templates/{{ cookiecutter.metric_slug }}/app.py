from {{ cookiecutter.metric_slug }} import {{ cookiecutter.metric_class_name }}
from evaluate.utils import launch_gradio_widget


metric = {{ cookiecutter.metric_class_name }}()
launch_gradio_widget(metric)
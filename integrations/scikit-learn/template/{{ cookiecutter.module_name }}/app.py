import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("{{ cookiecutter.namespace }}/{{ cookiecutter.module_name }}")
launch_gradio_widget(module)
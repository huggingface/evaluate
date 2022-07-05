import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("{{ cookiecutter.namespace }}/{{ cookiecutter.module_slug }}")
launch_gradio_widget(module)
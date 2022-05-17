from {{ cookiecutter.module_slug }} import {{ cookiecutter.module_class_name }}
from evaluate.utils import launch_gradio_widget


module = {{ cookiecutter.module_class_name }}()
launch_gradio_widget(module)
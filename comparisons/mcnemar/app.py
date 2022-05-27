import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("mcnemar", module_type="comparison")
launch_gradio_widget(module)

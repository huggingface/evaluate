import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("exact_match", module_type="comparison")
launch_gradio_widget(module)

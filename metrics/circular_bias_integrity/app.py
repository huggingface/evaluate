import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("circular_bias_integrity")
launch_gradio_widget(module)

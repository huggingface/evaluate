import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("exact_match")
launch_gradio_widget(module)

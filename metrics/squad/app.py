import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("squad")
launch_gradio_widget(module)

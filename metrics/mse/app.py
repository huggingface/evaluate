import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("mse")
launch_gradio_widget(module)

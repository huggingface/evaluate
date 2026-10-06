import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("ece")
launch_gradio_widget(module)

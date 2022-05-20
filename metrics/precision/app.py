import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("precision")
launch_gradio_widget(module)

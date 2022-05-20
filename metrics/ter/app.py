import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("ter")
launch_gradio_widget(module)

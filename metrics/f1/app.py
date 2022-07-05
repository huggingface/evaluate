import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("f1")
launch_gradio_widget(module)

import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("r2")
launch_gradio_widget(module)

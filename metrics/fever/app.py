import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("fever")
launch_gradio_widget(module)

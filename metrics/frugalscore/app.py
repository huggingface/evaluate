import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("frugalscore")
launch_gradio_widget(module)

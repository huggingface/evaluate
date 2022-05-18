import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("chrf")
launch_gradio_widget(module)

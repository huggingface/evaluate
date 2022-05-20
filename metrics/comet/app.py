import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("comet")
launch_gradio_widget(module)

import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("rouge")
launch_gradio_widget(module)

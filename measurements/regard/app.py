import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("regard")
launch_gradio_widget(module)

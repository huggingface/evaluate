import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("smape")
launch_gradio_widget(module)

import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("matthews_correlation")
launch_gradio_widget(module)

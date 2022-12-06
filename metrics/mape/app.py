import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("mape")
launch_gradio_widget(module)

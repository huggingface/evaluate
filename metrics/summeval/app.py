import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("summeval")
launch_gradio_widget(module)

import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("cuad")
launch_gradio_widget(module)

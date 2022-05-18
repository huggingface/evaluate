import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("squad_v2")
launch_gradio_widget(module)

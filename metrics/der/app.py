import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("der")
launch_gradio_widget(module)

import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("anls")
launch_gradio_widget(module)

import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("glue")
launch_gradio_widget(module)

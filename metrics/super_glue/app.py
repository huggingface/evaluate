import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("super_glue")
launch_gradio_widget(module)

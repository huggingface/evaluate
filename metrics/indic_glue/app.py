import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("indic_glue")
launch_gradio_widget(module)

import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("coval")
launch_gradio_widget(module)

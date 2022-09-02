import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("nist_mt")
launch_gradio_widget(module)

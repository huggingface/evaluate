import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("charcut_mt")
launch_gradio_widget(module)

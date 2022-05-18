import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("xtreme_s")
launch_gradio_widget(module)

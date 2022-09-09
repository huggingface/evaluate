import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("bary_score")
launch_gradio_widget(module)

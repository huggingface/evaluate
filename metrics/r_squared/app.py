import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("r_squared")
launch_gradio_widget(module)

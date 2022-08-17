import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("bold", "profession")
launch_gradio_widget(module)

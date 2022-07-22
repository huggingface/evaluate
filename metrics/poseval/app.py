import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("poseval")

launch_gradio_widget(module)

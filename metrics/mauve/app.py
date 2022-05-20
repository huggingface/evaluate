import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("mauve")
launch_gradio_widget(module)

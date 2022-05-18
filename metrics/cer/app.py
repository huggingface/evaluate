import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("cer")
launch_gradio_widget(module)

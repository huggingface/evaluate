import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("sari")
launch_gradio_widget(module)

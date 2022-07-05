import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("accuracy")
launch_gradio_widget(module)

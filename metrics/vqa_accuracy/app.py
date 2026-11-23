import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("vqa_accuracy")
launch_gradio_widget(module)
import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("confusion_matrix")
launch_gradio_widget(module)

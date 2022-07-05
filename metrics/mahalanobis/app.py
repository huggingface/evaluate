import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("mahalanobis")
launch_gradio_widget(module)

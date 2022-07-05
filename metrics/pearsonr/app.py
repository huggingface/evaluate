import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("pearsonr")
launch_gradio_widget(module)

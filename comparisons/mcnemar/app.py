import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("mcnemar", type="comparison")
launch_gradio_widget(module)

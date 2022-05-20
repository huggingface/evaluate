import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("spearmanr")
launch_gradio_widget(module)

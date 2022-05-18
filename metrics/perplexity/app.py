import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("perplexity")
launch_gradio_widget(module)

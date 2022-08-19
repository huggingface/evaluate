import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("toxicity")
launch_gradio_widget(module)

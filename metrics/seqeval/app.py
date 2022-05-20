import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("seqeval")
launch_gradio_widget(module)

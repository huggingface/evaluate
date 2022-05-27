import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("text_duplicates")
launch_gradio_widget(module)

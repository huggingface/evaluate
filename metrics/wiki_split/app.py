import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("wiki_split")
launch_gradio_widget(module)

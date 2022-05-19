import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("word_length", type="measurement")
launch_gradio_widget(module)

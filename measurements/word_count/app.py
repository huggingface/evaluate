import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("word_count", type="measurement")
launch_gradio_widget(module)

import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("perplexity", module_type="measurement")
launch_gradio_widget(module)

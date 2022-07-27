import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("label_distribution", module_type="measurement")
launch_gradio_widget(module)

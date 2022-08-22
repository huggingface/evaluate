import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("wilcoxon", module_type="comparison")
launch_gradio_widget(module)

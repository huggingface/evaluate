import evaluate
from evaluate.utils import launch_gradio_widget

module = evaluate.load("r_squared", module_type="metric")
launch_gradio_widget(module)

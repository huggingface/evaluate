import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("unnati/kendall_tau_distance")
launch_gradio_widget(module)

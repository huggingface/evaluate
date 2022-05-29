import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("lvwerra/trec_eval")
launch_gradio_widget(module)
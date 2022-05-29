import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("lvwerra/rl_reliability", "online")
launch_gradio_widget(module)

import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("competition_math")
launch_gradio_widget(module)

import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("glue", "sst2")
launch_gradio_widget(module)

import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("sacrebleu")
launch_gradio_widget(module)

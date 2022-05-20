import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("roc_auc")
launch_gradio_widget(module)

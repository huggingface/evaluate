import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("mean_iou")
launch_gradio_widget(module)

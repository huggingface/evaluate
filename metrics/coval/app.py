import sys

import evaluate
from evaluate.utils import launch_gradio_widget


sys.path = [p for p in sys.path if p != "/home/user/app"]
module = evaluate.load("coval")
sys.path = ["/home/user/app"] + sys.path

launch_gradio_widget(module)

import sys
from pathlib import Path

import gradio as gr
import evaluate
from evaluate import parse_readme

metric = evaluate.load(str(Path(sys.path[0])))


def compute_clip_score(image, text):
    results = metric.compute(predictions=[text], images=[image])
    per_pair = results["per_image_clip_score"][0]
    return per_pair


iface = gr.Interface(
    fn=compute_clip_score,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(lines=2, placeholder="Enter caption here..."),
    ],
    outputs=gr.Number(label="CLIP Score"),
    title="CLIPScore Evaluator",
    description=(
        "Evaluate alignment between an image and a caption using CLIPScore "
        "(Hessel et al., EMNLP 2021). Score range is 0 to 2.5."
    ),
    examples=[
        # positive pairs
        [
            "https://images.unsplash.com/photo-1720539222585-346e73f01536",
            "A cat sitting on a couch",
        ],
        [
            "https://images.unsplash.com/photo-1694253987647-4eebcf679974",
            "A scenic view of mountains during sunset",
        ],
        # negative pairs
        [
            "https://images.unsplash.com/photo-1720539222585-346e73f01536",
            "An airplane flying over the ocean",
        ],
        [
            "https://images.unsplash.com/photo-1694253987647-4eebcf679974",
            "A dog playing in the snow",
        ],
    ],
    examples_per_page=4,
    article=parse_readme(Path(sys.path[0]) / "README.md"),
)

iface.launch()
import sys
from pathlib import Path

import gradio as gr

import evaluate
from evaluate import parse_readme


metric = evaluate.load("sunhill/clip_score")


def compute_clip_score(image, text):
    results = metric.compute(predictions=[text], references=[image])
    return results["clip_score"]


iface = gr.Interface(
    fn=compute_clip_score,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(lines=2, placeholder="Enter text here..."),
    ],
    outputs=gr.Number(label="CLIP Score"),
    title="CLIP Score Evaluator",
    description="Evaluate the alignment between an image and a text using CLIP Score.",
    examples=[
        [
            "https://images.unsplash.com/photo-1720539222585-346e73f01536",
            "A cat sitting on a couch",
        ],
        [
            "https://images.unsplash.com/photo-1694253987647-4eebcf679974",
            "A scenic view of mountains during sunset",
        ],
    ],
    article=parse_readme(Path(sys.path[0]) / "README.md"),
)

iface.launch()

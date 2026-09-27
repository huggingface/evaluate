---
title: CLIP Score
tags:
- evaluate
- metric
description: "CLIPScore is a reference-free evaluation metric for image captioning that measures the alignment between images and their corresponding text descriptions."
sdk: gradio
sdk_version: 5.45.0
app_file: app.py
pinned: false
---

# Metric Card for CLIP Score

***Module Card Instructions:*** *This module calculates CLIPScore, a reference-free evaluation metric for image captioning.*

## Metric Description

CLIPScore is a reference-free evaluation metric for image captioning that measures the alignment between images and their corresponding text descriptions. It leverages the CLIP (Contrastive Language-Image Pretraining) model to compute a similarity score between the visual and textual modalities.

## How to Use

To use the CLIPScore metric, you need to provide a list of text predictions and a list of images. The metric will compute the CLIPScore for each pair of image and text.

### Inputs

- **predictions** *(string): A list of text predictions to score. Each prediction should be a string.*
- **references** *(PIL.Image.Image): A list of images to score against. Each image should be a PIL image.*

### Output Values

The CLIPScore metric outputs a dictionary with a single key-value pair:

- **clip_score** *(float)*: The average CLIPScore across all provided image-text pairs. The score ranges from -1 to 1, where higher scores indicate better alignment between the image and text.

### Examples

```python
from PIL import Image
import evaluate

metric = evaluate.load("sunhill/clip_score")
predictions = ["A cat sitting on a windowsill.", "A dog playing with a ball."]
references = [Image.open("cat.jpg"), Image.open("dog.jpg")]
results = metric.compute(predictions=predictions, references=references)
print(results)
# Output: {'clip_score': 0.85}
```

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2104-08718,
    author       = {Jack Hessel and
                    Ari Holtzman and
                    Maxwell Forbes and
                    Ronan Le Bras and
                    Yejin Choi},
    title        = {CLIPScore: {A} Reference-free Evaluation Metric for Image Captioning},
    journal      = {CoRR},
    volume       = {abs/2104.08718},
    year         = {2021},
    url          = {https://arxiv.org/abs/2104.08718},
    eprinttype   = {arXiv},
    eprint       = {2104.08718},
    timestamp    = {Sat, 29 Apr 2023 10:09:27 +0200},
    biburl       = {https://dblp.org/rec/journals/corr/abs-2104-08718.bib},
    bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Further References

- [clip-score](https://github.com/Taited/clip-score)

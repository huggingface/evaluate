where R is the set of human-written reference captions.

## How to Use

```python
from PIL import Image
import evaluate

metric = evaluate.load("path/to/clip_score")

# CLIPScore (reference-free)
predictions = ["A cat sitting on a couch", "A dog running in the park"]
images = [Image.open("cat.jpg"), Image.open("dog.jpg")]

results = metric.compute(predictions=predictions, images=images)
print(results["clip_score"])                # corpus-level average
print(results["per_image_clip_score"])      # per-pair scores

# RefCLIPScore (reference-augmented)
references = [
    ["a cat on a sofa", "a cat resting indoors"],
    ["a dog playing outside", "a dog in the park"],
]
results = metric.compute(predictions=predictions, images=images, references=references)
print(results["ref_clip_score"])                # corpus-level RefCLIPScore
print(results["per_image_ref_clip_score"])      # per-pair RefCLIPScores
```

### Inputs

- **predictions** (`list of str`): Candidate captions. The "A photo depicts" prefix is added
  automatically as recommended by the paper — do not add it yourself.
- **images** (`list of PIL.Image`): Images to score against. One image per caption.
- **references** (`list of list of str`, optional): Human-written reference captions per image,
  used to compute RefCLIPScore. Each entry is a list of reference strings for that image.

### Output Values

- **clip_score** (`float`): Average CLIPScore across all image-caption pairs (corpus-level).
  Range is 0 to 2.5, where higher means better alignment between image and caption.
- **per_image_clip_score** (`list of float`): Individual CLIPScore for each image-caption pair.
  In our validation on Flickr8k, correct image-caption pairs clustered around 0.70–0.95
  while mismatched pairs (wrong caption for an image) clustered around 0.27–0.59,
  showing the metric cleanly separates good from bad captions.
- **ref_clip_score** (`float`): Average RefCLIPScore across all pairs. Only returned when
  `references` are provided.
- **per_image_ref_clip_score** (`list of float`): Individual RefCLIPScore per pair. Only
  returned when `references` are provided.

### Score Range

Scores range from **0 to 2.5**. Negative cosine similarities are clipped to 0 before
scaling by w=2.5. A score of 0 means no alignment; scores above ~0.7 generally indicate
strong alignment between image and caption.

Note: do not compare scores across different CLIP model sizes — the rescaling factor w=2.5
was calibrated for ViT-B/32 specifically.

## Validation

We validate this implementation by replicating the human correlation experiments from the
original paper on the full Flickr8k-Expert dataset (16,992 image-caption pairs with expert
human quality ratings from 1 to 4):

| Metric | This implementation | Paper (Hessel et al., 2021) |
|--------|--------------------|-----------------------------|
| Kendall tau-c | **51.5** | 51.2 |
| Kendall tau-b | **51.1** | ~51.1 |

Results are within 0.3 points of the paper, consistent with minor floating-point
differences across hardware (the paper ran on GPU; small CPU/GPU precision differences
in CLIP are documented in the original repository).

## Model

Uses `openai/clip-vit-base-patch32` (ViT-B/32, 512-dimensional embeddings) as specified
in the original paper. Automatically uses GPU if available, CPU otherwise.

The "A photo depicts" prefix is hard-coded for text inputs as recommended by the paper
for best correlation with human judgments. Changing this prefix will produce results that
are not comparable to scores reported in the paper.

## Citation

```bibtex
@inproceedings{hessel-etal-2021-clipscore,
    title = "{CLIPS}core: A Reference-free Evaluation Metric for Image Captioning",
    author = "Hessel, Jack and
      Holtzman, Ari and
      Forbes, Maxwell and
      Le Bras, Ronan and
      Choi, Yejin",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    year = "2021",
    url = "https://arxiv.org/abs/2104.08718",
}
```

## Further References

- [CLIPScore paper](https://arxiv.org/abs/2104.08718)
- [Official CLIPScore implementation](https://github.com/jmhessel/clipscore) (Hessel et al.)
- [CLIP model](https://huggingface.co/openai/clip-vit-base-patch32)
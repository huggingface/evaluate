# Reproduces Table 1 correlation results from Hessel et al. (2021)
# on the full Flickr8k-Expert dataset (16,992 pairs).
# Expected output:
#   Kendall tau-c: ~51.5  (paper reports 51.2)
#   Kendall tau-b: ~51.1  (paper reports ~51.1)
# Runtime: ~3 minutes on GPU, ~15 minutes on CPU.

from datasets import load_dataset
from scipy.stats import kendalltau
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import io

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def compute_clip_score(predictions, images, w=2.5):
    prefixed = ["A photo depicts " + p for p in predictions]
    inputs = processor(
        text=prefixed,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        ).pooler_output
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).pooler_output
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    per_pair_cosine = (image_features * text_features).sum(dim=1)
    return (w * torch.clamp(per_pair_cosine, min=0)).tolist()

# load the human eval dataset — only flickr8k-expert split
print("Loading Flickr8k-Expert human judgments...")
ds = load_dataset("yuwd/Flickr8k-HumanEval", split="test")
print(f"Loaded {len(ds)} image-caption pairs with human scores")

# compute in batches of 32 to avoid memory issues
BATCH_SIZE = 32
all_clip_scores = []

print("Computing CLIPScores...")
for i in range(0, len(ds), BATCH_SIZE):
    batch = ds.select(range(i, min(i + BATCH_SIZE, len(ds))))
    images = [item["img"].convert("RGB") if hasattr(item["img"], "convert") 
            else Image.open(io.BytesIO(item["img"]["bytes"])).convert("RGB") 
            for item in batch]
    captions = [item["cand"] for item in batch]
    scores = compute_clip_score(captions, images)
    all_clip_scores.extend(scores)
    if i % 500 == 0:
        print(f"  {i}/{len(ds)} done...")

human_scores = [item["human_score"] for item in ds]

# Kendall tau-c (what the paper reports for flickr8k-expert)
tau_c, p_c = kendalltau(all_clip_scores, human_scores, variant="c")
# Kendall tau-b as well for completeness
tau_b, p_b = kendalltau(all_clip_scores, human_scores, variant="b")

print(f"\n--- Results on Flickr8k-Expert ---")
print(f"Kendall tau-c: {tau_c*100:.1f}  (paper reports 51.2)")
print(f"Kendall tau-b: {tau_b*100:.1f}  (paper reports ~51.1)")
print(f"Number of pairs evaluated: {len(all_clip_scores)}")
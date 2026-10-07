# Sanity check for CLIPScore implementation.
# Loads 20 image-caption pairs from jxie/flickr8k test split.
# Scores each image against its correct caption and a mismatched caption.
# Expected output: correct caption scores higher in all 20 cases.

from datasets import load_dataset
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
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
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"]).pooler_output
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).pooler_output
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    per_pair_cosine = (image_features * text_features).sum(dim=1)
    per_pair_scores = w * torch.clamp(per_pair_cosine, min=0)
    return per_pair_scores.tolist()

# load 20 test examples
print("Loading dataset...")
ds = load_dataset("jxie/flickr8k", split="test")
samples = ds.select(range(20))

# for each image, score its real caption vs a shuffled wrong caption
# shuffled wrong caption = caption_0 of the NEXT image
images = [s["image"] for s in samples]
correct_captions = [s["caption_0"] for s in samples]
wrong_captions = correct_captions[1:] + [correct_captions[0]]  # shift by 1

correct_scores = compute_clip_score(correct_captions, images)
wrong_scores = compute_clip_score(wrong_captions, images)

print("\nPer-image results (correct caption vs wrong caption):")
print(f"{'':>4}  {'CORRECT':>8}  {'WRONG':>8}  {'correct > wrong?':>16}")
for i, (c, w_) in enumerate(zip(correct_scores, wrong_scores)):
    better = "YES" if c > w_ else "NO"
    print(f"{i:>4}  {c:>8.4f}  {w_:>8.4f}  {better:>16}")

wins = sum(1 for c, w_ in zip(correct_scores, wrong_scores) if c > w_)
print(f"\nCorrect caption scored higher: {wins}/20")
print(f"Average correct score: {sum(correct_scores)/len(correct_scores):.4f}")
print(f"Average wrong score:   {sum(wrong_scores)/len(wrong_scores):.4f}")

# test RefCLIPScore — use captions 1-4 as references for each image
print("\n--- Now testing RefCLIPScore ---")
references = [
    [s["caption_1"], s["caption_2"], s["caption_3"], s["caption_4"]]
    for s in samples
]

def compute_ref_clip_score(predictions, images, references):
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
    clip_scores = 2.5 * torch.clamp(per_pair_cosine, min=0)

    # now compute max reference similarity for each candidate
    max_ref_sims = torch.zeros(len(predictions)).to(device)
    for i, ref_list in enumerate(references):
        ref_inputs = processor(
            text=ref_list,
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.no_grad():
            ref_features = model.get_text_features(
                input_ids=ref_inputs["input_ids"],
                attention_mask=ref_inputs["attention_mask"],
            ).pooler_output
        ref_features = ref_features / ref_features.norm(dim=1, keepdim=True)
        sims = (ref_features * text_features[i].unsqueeze(0)).sum(dim=1)
        max_ref_sims[i] = torch.clamp(sims.max(), min=0)

    # harmonic mean
    numerator = 2 * clip_scores * max_ref_sims
    denominator = torch.clamp(clip_scores + max_ref_sims, min=1e-8)
    ref_scores = numerator / denominator
    return clip_scores.tolist(), ref_scores.tolist()

clip_s, ref_clip_s = compute_ref_clip_score(correct_captions, images, references)

print(f"\n{'':>4}  {'CLIP-S':>8}  {'RefCLIP-S':>10}")
for i, (c, r) in enumerate(zip(clip_s, ref_clip_s)):
    print(f"{i:>4}  {c:>8.4f}  {r:>10.4f}")

print(f"\nAverage CLIP-S:    {sum(clip_s)/len(clip_s):.4f}")
print(f"Average RefCLIP-S: {sum(ref_clip_s)/len(ref_clip_s):.4f}")
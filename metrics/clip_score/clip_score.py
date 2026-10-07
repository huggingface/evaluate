# Copyright 2024 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLIPScore: a reference-free evaluation metric for image captioning."""

import datasets
import torch
from transformers import CLIPModel, CLIPProcessor

import evaluate
from evaluate.utils.logging import get_logger

logger = get_logger(__name__)

_CITATION = """\
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
"""

_DESCRIPTION = """\
CLIPScore is a reference-free evaluation metric for image captioning.
It measures the alignment between an image and a candidate caption using
CLIP embeddings, without requiring human-written reference captions.
Formula: CLIP-S(c, v) = 2.5 * max(cos(c, v), 0)
where c is the text embedding and v is the image embedding.
"""

_KWARGS_DESCRIPTION = """
Computes CLIPScore to evaluate alignment between images and text captions.

Args:
    predictions: list of candidate captions to score. Each should be a string.
    images: list of images to score against. Each should be a PIL image.

Returns:
    clip_score: average CLIPScore across all image-caption pairs (corpus-level).
    per_image_clip_score: list of CLIPScores for each individual pair.

Examples:
    >>> from PIL import Image
    >>> metric = evaluate.load("path/to/clip_score")
    >>> results = metric.compute(
    ...     predictions=["A cat sitting on a couch"],
    ...     images=[Image.open("cat.jpg")]
    ... )
    >>> print(results)
    {'clip_score': 0.8234, 'per_image_clip_score': [0.8234]}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CLIPScore(evaluate.Metric):
    """CLIPScore metric."""

    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "images": datasets.Image(),
                }
            ),
            homepage="https://arxiv.org/abs/2104.08718",
            codebase_urls=["https://github.com/huggingface/evaluate"],
            reference_urls=["https://arxiv.org/abs/2104.08718"],
        )

    def _download_and_prepare(self, dl_manager):
        """Load CLIP model and processor onto the right device."""
        logger.info("Loading CLIP ViT-B/32 model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def _compute(self, predictions, images, references=None, w=2.5):
        """
        predictions: list of candidate captions (strings)
        images: list of PIL images
        references: optional list of lists of reference captions
                    e.g. [["a dog runs", "a brown dog"], ["a cat sits", "cat on mat"]]
        w: rescaling factor from paper (default 2.5)
        """
        prefixed = ["A photo depicts " + p for p in predictions]

        inputs = self.processor(
            text=prefixed,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(
                pixel_values=inputs["pixel_values"]
            ).pooler_output
            text_features = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).pooler_output

        # normalize
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # per-pair cosine similarity, shape [batch_size]
        per_pair_cosine = (image_features * text_features).sum(dim=1)

        # CLIP-S(c, v) = w * max(cos(c, v), 0)
        clip_scores = w * torch.clamp(per_pair_cosine, min=0)

        result = {
            "clip_score": clip_scores.mean().item(),
            "per_image_clip_score": clip_scores.tolist(),
        }

        # RefCLIPScore — only computed if references are provided
        if references is not None:
            ref_clip_scores = self._compute_ref_clip_scores(
                text_features, references, clip_scores
            )
            result["ref_clip_score"] = ref_clip_scores.mean().item()
            result["per_image_ref_clip_score"] = ref_clip_scores.tolist()

        return result


    def _compute_ref_clip_scores(self, candidate_features, references, clip_scores):
        """
        candidate_features: normalized text embeddings, shape [batch_size, 512]
        references: list of lists of reference caption strings
        clip_scores: already computed CLIP-S scores, shape [batch_size]
        
        For each candidate, computes:
        max_ref_sim = max over all references r of max(cos(c, r), 0)
        RefCLIP-S = harmonic_mean(CLIP-S, max_ref_sim)
        """
        batch_size = candidate_features.shape[0]
        max_ref_sims = torch.zeros(batch_size).to(self.device)

        for i, ref_list in enumerate(references):
            # encode all references for this candidate
            ref_inputs = self.processor(
                text=ref_list,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                ref_features = self.model.get_text_features(
                    input_ids=ref_inputs["input_ids"],
                    attention_mask=ref_inputs["attention_mask"],
                ).pooler_output

            # normalize reference features
            ref_features = ref_features / ref_features.norm(dim=1, keepdim=True)

            # cosine similarity between candidate i and each reference
            # candidate_features[i] is shape [512]
            # ref_features is shape [num_refs, 512]
            sims = (ref_features * candidate_features[i].unsqueeze(0)).sum(dim=1)

            # max(max_r cos(c, r), 0)
            max_ref_sims[i] = torch.clamp(sims.max(), min=0)

        # harmonic mean of CLIP-S and max_ref_sim
        # H-Mean(a, b) = 2ab / (a + b)
        # guard against division by zero
        numerator = 2 * clip_scores * max_ref_sims
        denominator = clip_scores + max_ref_sims
        denominator = torch.clamp(denominator, min=1e-8)
        ref_clip_scores = numerator / denominator

        return ref_clip_scores
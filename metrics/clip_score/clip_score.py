# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""This module calculates CLIPScore, a reference-free evaluation metric for image captioning."""

import datasets
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

import evaluate
from evaluate.utils.logging import get_logger


logger = get_logger(__name__)

_CITATION = """\
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
"""

_DESCRIPTION = """\
This new module is designed to calculate CLIPScore, a reference-free evaluation metric for image captioning.
"""


_KWARGS_DESCRIPTION = """
Computes CLIPScore to evaluate the alignment between an image and a text.
Args:
    predictions: list of text predictions to score. Each prediction
        should be a string.
    images: list of images to score against. Each image should be a PIL image.
Returns:
    clip_score: CLIPScore between the image and the text.
Examples:
    >>> metric = evaluate.load("sunhill/clip_score")
    >>> results = metric.compute(predictions=["A cat sitting on a couch."], images=[PIL_image])
    >>> print(results)
    {'clip_score': 0.2076}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CLIPScore(evaluate.Metric):
    """CLIPScore metric."""

    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Image(),
                }
            ),
            # Homepage of the module for documentation
            homepage="https://huggingface.co/spaces/sunhill/clip_score",
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/Taited/clip-score"],
            reference_urls=["https://arxiv.org/abs/2104.08718"],
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        logger.info("Downloading and preparing CLIP ViT-B/32 model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def _compute(self, predictions, references):
        """Returns the scores"""
        refer = self.processor(text=None, images=references, return_tensors="pt", padding=True)
        pred = self.tokenizer(predictions, return_tensors="pt", padding=True)

        refer_features = self.model.get_image_features(**refer)
        pred_features = self.model.get_text_features(**pred)

        refer_features = refer_features / refer_features.norm(dim=1, keepdim=True)
        pred_features = pred_features / pred_features.norm(dim=1, keepdim=True)
        clip_score = (refer_features * pred_features).sum().item()
        return {"clip_score": clip_score / refer_features.shape[0]}

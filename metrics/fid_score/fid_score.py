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

"""fid score metric."""
import numpy as np
from scipy.linalg import sqrtm

import evaluate


_CITATION = """\
@inproceedings{heusel2017gans,
  title={GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium},
  author={Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6626--6637},
  year={2017}
}
"""

_DESCRIPTION = """\
The Frechet Inception Distance (FID) is a metric used to evaluate the quality of generated images from generative adversarial networks (GANs). It measures the similarity between the feature representations of real and generated images.

FID is calculated by first extracting feature vectors from a pre-trained Inception-v3 network for both the real and generated images. Then, it computes the mean and covariance matrix of these feature vectors for each set. Finally, it calculates the Fréchet distance between these multivariate Gaussian distributions.

A lower FID score indicates a higher similarity between the real and generated images, suggesting better performance of the GAN in generating realistic images.

For further details, please refer to the paper:
"GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
by Heusel et al., presented at the Advances in Neural Information Processing Systems (NeurIPS) conference in 2017.
"""

_KWARGS_DESCRIPTION = """
Computes FID score between two sets of features.
Args:
    real_features: numpy array of feature vectors extracted from real images.
    fake_features: numpy array of feature vectors extracted from generated images.
Returns:
    (float): the Frechet Inception Distance (FID) score.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class FID(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=None,
            reference_urls=[],
        )

    def _compute(self, real_images, generated_images):
        real_images = real_images.reshape(real_images.shape[0] * real_images.shape[1], real_images.shape[2])
        generated_images = generated_images.reshape(
            generated_images.shape[0] * generated_images.shape[1], generated_images.shape[2]
        )
        mu_real = np.mean(real_images, axis=0)
        sigma_real = np.cov(real_images, rowvar=False)

        mu_generated = np.mean(generated_images, axis=0)
        sigma_generated = np.cov(generated_images, rowvar=False)

        mean_diff = mu_real - mu_generated
        mean_diff_squared = np.dot(mean_diff, mean_diff)

        cov_mean = sqrtm(sigma_real.dot(sigma_generated))

        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real

        fid = mean_diff_squared + np.trace(sigma_real + sigma_generated - 2 * cov_mean)
        return fid

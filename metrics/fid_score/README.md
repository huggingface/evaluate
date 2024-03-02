---
title: fid_score
emoji: 🤗 
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: >-
The Frechet Inception Distance (FID) is a metric used to evaluate the quality of generated images from generative adversarial networks (GANs). It measures the similarity between the feature representations of real and generated images.

FID is calculated by first extracting feature vectors from a pre-trained Inception-v3 network for both the real and generated images. Then, it computes the mean and covariance matrix of these feature vectors for each set. Finally, it calculates the Fréchet distance between these multivariate Gaussian distributions.

A lower FID score indicates a higher similarity between the real and generated images, suggesting better performance of the GAN in generating realistic images.

For further details, please refer to the paper:
"GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
by Heusel et al., presented at the Advances in Neural Information Processing Systems (NeurIPS) conference in 2017.
---

# Metric Card for fid_score

## Metric description

The Frechet Inception Distance (FID) is a metric used to evaluate the quality of generated images from generative adversarial networks (GANs). It measures the similarity between the feature representations of real and generated images.

FID is calculated by first extracting feature vectors from a pre-trained Inception-v3 network for both the real and generated images. Then, it computes the mean and covariance matrix of these feature vectors for each set. Finally, it calculates the Fréchet distance between these multivariate Gaussian distributions.

A lower FID score indicates a higher similarity between the real and generated images, suggesting better performance of the GAN in generating realistic images.


## How to use 

The metric takes two inputs: references (a list of references for each speech input) and predictions (a list of transcriptions to score).

```python
from evaluate import load
fid = load("fid_score")
fid_score = fid.compute(real_images=real_images, generated_images=generated_images)
```
## Output values

This metric outputs a float which is a single floating-point number representing the dissimilarity between the distributions of real and generated images.
```
print(fid_score)
73.65338799020432
```

The **lower** the fid_score value, the **better** the performance of the GAN.

### Values from popular papers
## Examples 

Perfect match between two same GANs images:

```python
from evaluate import load
fid = load("fid_score")
im1 = cv2.imread("gans1.png")
im2 = cv2.imread("gans1.png")
fid_score = fid.compute(real_images=im1, generated_images=im2)
print(fid_score)
5.020410753786564e-10
```

Partial match between two GANs images:

```python
from evaluate import load
fid = load("fid_score")
im1 = cv2.imread("gans1.png")
im2 = cv2.imread("gans2.png")
fid_score = fid.compute(real_images=im1, generated_images=im2)
print(fid_score)
73.65338799020432
```

No match between two random images:

```python
from evaluate import load
fid = load("fid_score")
im1 = cv2.imread("gans1.png")
im2 = cv2.imread("random.png")
fid_score = fid.compute(real_images=im1, generated_images=im2)
print(fid_score)
10991.947362765806
```




## Limitations and bias

The Frechet Inception Distance (FID) metric, while widely used for assessing the quality of generated images, has limitations and biases. It relies on features extracted from a pretrained Inception network, making it sensitive to network biases and changes in architecture or training procedures. FID may not capture all aspects of image quality, especially semantic differences or specific image features, and its interpretation lacks detailed insights into image generation. Additionally, FID is computationally complex and lacks a universally agreed-upon threshold for acceptable scores, requiring careful consideration alongside qualitative evaluation methods. Despite these limitations, FID remains valuable for comparing generative models, particularly in the context of GANs, but should be used judiciously alongside other evaluation metrics.


## Citation


```bibtex
@inproceedings{heusel2017gans,
  title={GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium},
  author={Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6626--6637},
  year={2017}
}
```

## Further References 

- [Fréchet inception distance -- Wikipedia](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)

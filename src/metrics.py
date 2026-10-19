from collections.abc import Iterable

import numpy as np
import torch
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity
from torch import nn
from torch.nn import functional as F
from torchvision.models import Inception_V3_Weights, inception_v3


class InceptionFeatures(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.fc = nn.Identity()
        self.model.eval().to(device)
        self.device = device
        self.register_buffer(
            "mean",
            torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1),
        )

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.add(1).div(2).clamp(0, 1)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        features = self.model((images - self.mean) / self.std)
        return features.flatten(1)


def calculate_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    if real_features.ndim != 2 or generated_features.ndim != 2:
        raise ValueError("FID features must be two-dimensional arrays")
    if real_features.shape[1] != generated_features.shape[1]:
        raise ValueError("Real and generated FID features must have the same width")
    if min(real_features.shape[0], generated_features.shape[0]) < 2:
        raise ValueError("FID requires at least two real and generated images")
    real_mean, real_cov = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    generated_mean, generated_cov = (
        np.mean(generated_features, axis=0),
        np.cov(generated_features, rowvar=False),
    )
    epsilon = 1e-6
    identity = np.eye(real_cov.shape[0], dtype=real_cov.dtype)
    covariance_mean = sqrtm((real_cov + epsilon * identity) @ (generated_cov + epsilon * identity))
    if not np.isfinite(covariance_mean).all():
        covariance_mean = sqrtm(
            (real_cov + 1e-3 * identity) @ (generated_cov + 1e-3 * identity)
        )
    if not np.isfinite(covariance_mean).all():
        raise ValueError("FID covariance square root is not finite")
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    score = (
        np.sum((real_mean - generated_mean) ** 2)
        + np.trace(real_cov)
        + np.trace(generated_cov)
        - 2 * np.trace(covariance_mean)
    )
    return float(max(score, 0.0))


def calculate_ssim(real: torch.Tensor, generated: torch.Tensor) -> float:
    if real.shape != generated.shape or real.ndim != 4 or real.shape[1] != 3:
        raise ValueError("SSIM expects matching RGB image batches in NCHW format")
    scores = []
    for reference, prediction in zip(real, generated):
        reference_array = reference.detach().cpu().permute(1, 2, 0).numpy()
        prediction_array = prediction.detach().cpu().permute(1, 2, 0).numpy()
        scores.append(
            structural_similarity(
                reference_array,
                prediction_array,
                channel_axis=2,
                data_range=2.0,
            )
        )
    return float(np.mean(scores))


def collect_features(
    extractor: InceptionFeatures,
    image_batches: Iterable[torch.Tensor],
) -> np.ndarray:
    features = []
    for images in image_batches:
        features.append(extractor(images.to(extractor.device)).cpu().numpy())
    if not features:
        raise ValueError("Cannot calculate FID for an empty image set")
    return np.concatenate(features, axis=0)

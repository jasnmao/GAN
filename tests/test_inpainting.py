import torch

from src.masks import make_masks
from src.metrics import calculate_fid, calculate_ssim
from src.models import InpaintingGenerator, PatchDiscriminator


def test_masks_hide_pixels_and_keep_visible_pixels():
    for strategy in ("random", "rectangle"):
        masks = make_masks(3, 128, 128, strategy, "cpu")
        assert masks.shape == (3, 1, 128, 128)
        assert torch.all((masks == 0) | (masks == 1))
        assert torch.all(masks.sum(dim=(1, 2, 3)) > 0)
        assert torch.all(masks.sum(dim=(1, 2, 3)) < 128 * 128)


def test_generator_and_discriminator_shapes():
    generator = InpaintingGenerator(base_channels=8)
    discriminator = PatchDiscriminator(base_channels=8)
    images = torch.randn(2, 3, 128, 128)
    masks = torch.ones(2, 1, 128, 128)
    prediction = generator(images * masks, masks)
    assert prediction.shape == images.shape
    assert discriminator(prediction, masks).ndim == 4


def test_composite_preserves_observed_pixels_and_ssim_identity():
    images = torch.rand(2, 3, 32, 32) * 2 - 1
    masks = make_masks(2, 32, 32, "rectangle", "cpu")
    predictions = torch.zeros_like(images)
    completed = predictions * (1 - masks) + images * masks
    assert torch.equal(completed * masks, images * masks)
    assert calculate_ssim(images, images) == 1.0


def test_fid_is_zero_for_identical_feature_sets():
    features = torch.randn(8, 16).numpy()
    assert calculate_fid(features, features) < 1e-5

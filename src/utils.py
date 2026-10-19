import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.utils import make_grid, save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_generator(checkpoint_path: str | Path, device: torch.device):
    from .models import InpaintingGenerator

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = InpaintingGenerator(base_channels=checkpoint.get("base_channels", 64))
    model.load_state_dict(checkpoint["generator"])
    return model.to(device).eval(), checkpoint


def save_comparison_grid(
    originals: torch.Tensor,
    masks: torch.Tensor,
    completed: torch.Tensor,
    output_path: str | Path,
    count: int = 8,
) -> None:
    count = min(count, originals.shape[0])
    masked = originals * masks[:count] + (1 - masks[:count])
    rows = torch.stack(
        (originals[:count], masked, completed[:count], originals[:count]),
        dim=1,
    ).flatten(0, 1)
    grid = make_grid(rows, nrow=4, normalize=True, value_range=(-1, 1), padding=4)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)
    image = Image.open(output_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    labels = ("原图", "遮挡输入", "GAN修复", "参考原图")
    for column, label in enumerate(labels):
        draw.text((column * (originals.shape[-1] + 4) + 8, 4), label, fill="yellow")
    image.save(output_path)

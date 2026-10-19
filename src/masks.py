import random

import torch


def make_masks(
    batch_size: int,
    height: int,
    width: int,
    strategy: str,
    device: torch.device | str,
) -> torch.Tensor:
    """Return masks where 1 is visible and 0 is missing."""
    if strategy not in {"random", "rectangle"}:
        raise ValueError("strategy must be 'random' or 'rectangle'")
    if min(batch_size, height, width) < 1:
        raise ValueError("batch_size, height, and width must be positive")

    masks = torch.ones((batch_size, 1, height, width), device=device)
    for mask in masks:
        if strategy == "rectangle":
            box_height = random.randint(max(1, height // 4), max(1, height // 2))
            box_width = random.randint(max(1, width // 4), max(1, width // 2))
            top = random.randint(0, height - box_height)
            left = random.randint(0, width - box_width)
            mask[:, top : top + box_height, left : left + box_width] = 0
            continue

        for _ in range(random.randint(2, 5)):
            box_height = random.randint(max(1, height // 12), max(1, height // 3))
            box_width = random.randint(max(1, width // 12), max(1, width // 3))
            top = random.randint(0, height - box_height)
            left = random.randint(0, width - box_width)
            mask[:, top : top + box_height, left : left + box_width] = 0
    return masks

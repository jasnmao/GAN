import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from src.data import PROJECT_ROOT
from src.masks import make_masks
from src.utils import load_generator


def inpaint(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    generator, checkpoint = load_generator(args.checkpoint, device)
    image_size = checkpoint.get("image_size", 128)
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side)).resize((image_size, image_size))
        original = TF.to_tensor(image).mul(2).sub(1).unsqueeze(0).to(device)

    masks = make_masks(1, image_size, image_size, args.mask, device)
    with torch.inference_mode():
        prediction = generator(original * masks, masks)
        completed = prediction * (1 - masks) + original * masks
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(completed, output_path, normalize=True, value_range=(-1, 1))
    print(f"Repaired image saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inpaint a face image with a trained CelebA GAN.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "latest.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "inpainted.png")
    parser.add_argument("--mask", choices=("random", "rectangle"), default="rectangle")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    inpaint(parse_args())

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import CelebADataset, PROJECT_ROOT
from src.masks import make_masks
from src.models import InpaintingGenerator, PatchDiscriminator
from src.utils import set_seed


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = CelebADataset(args.data_root, "train", args.image_size, args.max_train_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("The training set must contain at least one full batch")

    generator = InpaintingGenerator(args.base_channels).to(device)
    discriminator = PatchDiscriminator(args.base_channels).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        totals = {"generator": 0.0, "discriminator": 0.0, "hole_l1": 0.0}
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for originals, _ in progress:
            originals = originals.to(device, non_blocking=True)
            strategy = args.mask_strategy
            if strategy == "mixed":
                strategy = "random" if torch.rand(()) < 0.5 else "rectangle"
            masks = make_masks(
                originals.shape[0],
                originals.shape[2],
                originals.shape[3],
                strategy,
                device,
            )
            masked = originals * masks
            missing = 1 - masks
            prediction = generator(masked, masks)
            completed = prediction * missing + originals * masks

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(originals, masks)
            fake_logits = discriminator(completed.detach(), masks)
            loss_d = torch.relu(1 - real_logits).mean() + torch.relu(1 + fake_logits).mean()
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            fake_logits = discriminator(completed, masks)
            hole_l1 = ((prediction - originals).abs() * missing).sum() / (
                missing.sum() * originals.shape[1] + 1e-8
            )
            loss_g = args.lambda_hole * hole_l1 - args.lambda_adv * fake_logits.mean()
            loss_g.backward()
            optimizer_g.step()

            totals["generator"] += loss_g.item()
            totals["discriminator"] += loss_d.item()
            totals["hole_l1"] += hole_l1.item()
            progress.set_postfix(g=f"{loss_g.item():.3f}", d=f"{loss_d.item():.3f}")

        batches = len(loader)
        print(
            f"epoch={epoch} "
            f"g={totals['generator'] / batches:.4f} "
            f"d={totals['discriminator'] / batches:.4f} "
            f"hole_l1={totals['hole_l1'] / batches:.4f}"
        )
        checkpoint = {
            "epoch": epoch,
            "base_channels": args.base_channels,
            "image_size": args.image_size,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "args": vars(args),
        }
        temporary_path = output_dir / "latest.tmp"
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(output_dir / "latest.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional GAN for CelebA inpainting.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-hole", type=float, default=100.0)
    parser.add_argument("--lambda-adv", type=float, default=0.01)
    parser.add_argument("--mask-strategy", choices=("mixed", "random", "rectangle"), default="mixed")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

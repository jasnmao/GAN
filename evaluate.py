import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import CelebADataset, PROJECT_ROOT
from src.masks import make_masks
from src.metrics import InceptionFeatures, calculate_fid, calculate_ssim
from src.utils import load_generator, save_comparison_grid, set_seed


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = CelebADataset(args.data_root, "test", args.image_size, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    generator, checkpoint = load_generator(args.checkpoint, device)
    if checkpoint.get("image_size", args.image_size) != args.image_size:
        raise ValueError(
            f"Checkpoint image size is {checkpoint['image_size']}; "
            f"requested evaluation size is {args.image_size}"
        )
    extractor = InceptionFeatures(device)
    results = {}
    output_dir = Path(args.output_dir)

    for strategy in ("random", "rectangle"):
        real_features, completed_features = [], []
        real_images, completed_images = [], []
        sample_originals, sample_masks, sample_completed = [], [], []
        generator.eval()
        for originals, _ in tqdm(loader, desc=f"Evaluating {strategy} masks"):
            originals = originals.to(device, non_blocking=True)
            masks = make_masks(
                originals.shape[0],
                originals.shape[2],
                originals.shape[3],
                strategy,
                device,
            )
            masked = originals * masks
            prediction = generator(masked, masks)
            completed = prediction * (1 - masks) + originals * masks
            real_features.append(extractor(originals).cpu())
            completed_features.append(extractor(completed).cpu())
            real_images.append(originals.cpu())
            completed_images.append(completed.cpu())
            if len(sample_originals) * args.batch_size < args.num_samples:
                sample_originals.append(originals.cpu())
                sample_masks.append(masks.cpu())
                sample_completed.append(completed.cpu())

        real_features_np = torch.cat(real_features).numpy()
        completed_features_np = torch.cat(completed_features).numpy()
        originals_all = torch.cat(real_images)
        completed_all = torch.cat(completed_images)
        scores = {
            "fid": calculate_fid(real_features_np, completed_features_np),
            "ssim": calculate_ssim(originals_all, completed_all),
            "samples": len(dataset),
        }
        results[strategy] = scores

        sample_originals_tensor = torch.cat(sample_originals)[: args.num_samples]
        sample_masks_tensor = torch.cat(sample_masks)[: args.num_samples]
        sample_completed_tensor = torch.cat(sample_completed)[: args.num_samples]
        save_comparison_grid(
            sample_originals_tensor,
            sample_masks_tensor,
            sample_completed_tensor,
            output_dir / f"{strategy}_comparison.png",
            args.num_samples,
        )
        print(f"{strategy}: FID={scores['fid']:.4f}, SSIM={scores['ssim']:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "metrics.json"
    result_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Metrics saved to {result_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GAN inpainting metrics across mask types.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "latest.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.max_samples < 2:
        parser.error("--max-samples must be at least 2 for FID")
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    return args


if __name__ == "__main__":
    evaluate(parse_args())

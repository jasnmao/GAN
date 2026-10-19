from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CelebADataset(Dataset):
    def __init__(
        self,
        data_root: str | Path = PROJECT_ROOT / "data",
        split: str = "train",
        image_size: int = 128,
        max_samples: int | None = None,
    ) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be 'train', 'valid', or 'test'")
        self.image_dir = Path(data_root) / "img_align_celeba"
        partition_file = Path(data_root) / "Eval" / "list_eval_partition.txt"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"CelebA image directory not found: {self.image_dir}")
        if not partition_file.is_file():
            raise FileNotFoundError(f"CelebA partition file not found: {partition_file}")

        wanted_partition = {"train": "0", "valid": "1", "test": "2"}[split]
        self.image_names = []
        with partition_file.open(encoding="utf-8") as partition:
            for line in partition:
                fields = line.split()
                if len(fields) >= 2 and fields[1] == wanted_partition:
                    self.image_names.append(fields[0])
        if not self.image_names:
            raise ValueError(f"No images found for CelebA split '{split}' in {partition_file}")
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive")
            self.image_names = self.image_names[:max_samples]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int):
        name = self.image_names[index]
        image_path = self.image_dir / name
        if not image_path.is_file():
            raise FileNotFoundError(f"CelebA image listed in partition file is missing: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BICUBIC,
            )
            tensor = TF.to_tensor(image).mul(2).sub(1)
        return tensor, name

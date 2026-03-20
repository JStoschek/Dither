import random
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


class DitherDataset(Dataset):
    """
    Paired dataset for:
      input  = grayscale image
      target = dithered image
    """
    def __init__(
        self,
        root="data",
        split="train",
        crop_size=128,
        random_crop=True,
        hflip=True,
        vflip=True,
        rotate=True,
    ):
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size

        self.random_crop = random_crop
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate

        self.gray_dir = self.root / "gray" / split
        self.dithered_dir = self.root / "dithered" / split

        self.samples = []
        for gray_path in sorted(self.gray_dir.iterdir()):
            if gray_path.suffix != ".png":
                continue

            dithered_name = gray_path.stem.replace("_gray", "_dithered") + gray_path.suffix
            dithered_path = self.dithered_dir / dithered_name

            if not dithered_path.exists():
                raise FileNotFoundError(
                    f"Missing matching dithered file for {gray_path.name} "
                    f"at {dithered_path}"
                )
            self.samples.append((gray_path, dithered_path))
        
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image pairs found in {self.gray_dir} and {self.dithered_dir}"
            )
        
    def __len__(self):
        return len(self.samples)

    def _paired_transform(self, gray, dithered):
        """
        Apply the exact same spatial transforms to both images.
        """
        # Make sure images are large enough for crop
        if gray.width < self.crop_size or gray.height < self.crop_size:
            new_w = max(gray.width, self.crop_size)
            new_h = max(gray.height, self.crop_size)

            gray = gray.resize((new_w, new_h), Image.BICUBIC)
            dithered = dithered.resize((new_w, new_h), Image.NEAREST)

        # Crop
        if self.random_crop:
            top = random.randint(0, gray.height - self.crop_size)
            left = random.randint(0, gray.width - self.crop_size)
        else:
            top = (gray.height - self.crop_size) // 2
            left = (gray.width - self.crop_size) // 2

        gray = TF.crop(gray, top, left, self.crop_size, self.crop_size)
        dithered = TF.crop(dithered, top, left, self.crop_size, self.crop_size)

        # Horizontal flip
        if self.hflip and random.random() < 0.5:
            gray = TF.hflip(gray)
            dithered = TF.hflip(dithered)

        # Vertical flip
        if self.vflip and random.random() < 0.5:
            gray = TF.vflip(gray)
            dithered = TF.vflip(dithered)

        # Random 90-degree rotations
        if self.rotate:
            k = random.randint(0, 3)  # 0,1,2,3 -> 0,90,180,270 degrees
            if k > 0:
                angle = 90 * k
                gray = TF.rotate(gray, angle)
                dithered = TF.rotate(dithered, angle)

        return gray, dithered

    def __getitem__(self, idx):
        gray_path, dithered_path = self.samples[idx]

        # Force grayscale
        gray = Image.open(gray_path).convert("L")
        dithered = Image.open(dithered_path).convert("L")

        gray, dithered = self._paired_transform(gray, dithered)

        # Convert to tensors in [0,1]
        gray = TF.to_tensor(gray)          # shape: (1, H, W), float32 in [0,1]
        dithered = TF.to_tensor(dithered)  # shape: (1, H, W), float32 in [0,1]

        # Make sure target is binary {0,1}
        dithered = (dithered > 0.5).float()

        return gray, dithered


if __name__ == "__main__":

    train_dataset = DitherDataset(
        root="data",
        split="train",
        crop_size=128,
        random_crop=True,
        hflip=True,
        vflip=True,
        rotate=True,
    )

    valid_dataset = DitherDataset(
        root="data",
        split="valid",
        crop_size=128,
        random_crop=False,  # center crop for validation
        hflip=False,
        vflip=False,
        rotate=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

    x, y = next(iter(train_loader))
    print(x.shape)  # (B, 1, 128, 128)
    print(y.shape)  # (B, 1, 128, 128)
    print(x.min().item(), x.max().item())
    print(y.min().item(), y.max().item())

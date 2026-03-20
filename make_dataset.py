"""
Download DIV2K high-resolution images and produce a paired
grayscale / Floyd-Steinberg dithered dataset for ML training.

Directory layout after running:
  data/
    DIV2K_train_HR/  raw extracted images
    DIV2K_valid_HR/
    gray/train/      grayscale versions at SCALE
    gray/valid/
    dithered/train/  Floyd-Steinberg dithered versions at SCALE
    dithered/valid/
"""

import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import numba
from PIL import Image
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
SCALE    = 1
DATA_DIR = Path("data")

_SPLITS = {
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}
# ──────────────────────────────────────────────────────────────────────────────


@numba.njit(cache=True)
def _fs_numba(buf, width, height):
    """Floyd-Steinberg dithering compiled by Numba"""
    for y in range(height):
        for x in range(width):
            old = buf[y * width + x]
            new = 255.0 if old >= 128.0 else 0.0
            buf[y * width + x] = new
            err = old - new

            if x + 1 < width:
                buf[y * width + x + 1]         += err * 7.0 / 16.0
            if y + 1 < height:
                if x - 1 >= 0:
                    buf[(y + 1) * width + x - 1] += err * 3.0 / 16.0
                buf[(y + 1) * width + x]         += err * 5.0 / 16.0
                if x + 1 < width:
                    buf[(y + 1) * width + x + 1] += err * 1.0 / 16.0
    return buf


def _dither(img: Image.Image) -> Image.Image:
    buf = np.array(img, dtype=np.float64).ravel()
    _fs_numba(buf, img.width, img.height)
    out = Image.new("L", (img.width, img.height))
    out.putdata(np.clip(buf, 0, 255).astype(np.uint8).tolist())
    return out


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb  = downloaded / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} MB", end="", flush=True)


def download_and_extract(split: str, url: str, raw_dir: Path):
    zip_path = DATA_DIR / f"DIV2K_{split}_HR.zip"

    if not zip_path.exists():
        print(f"Downloading {split} HR …")
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        print()
    else:
        print(f"  zip exists, skipping download")

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print(f"Extracting …")
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith((".png", ".jpg"))]
            for member in tqdm(members, desc="  extract", unit="file"):
                zf.extract(member, DATA_DIR)
    else:
        print(f"  already extracted, skipping")


def process_split(split: str, raw_dir: Path):
    gray_dir     = DATA_DIR / "gray"     / split
    dithered_dir = DATA_DIR / "dithered" / split
    gray_dir.mkdir(parents=True, exist_ok=True)
    dithered_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(raw_dir.rglob("*.png")) + sorted(raw_dir.rglob("*.jpg"))
    if not images:
        print(f"  no images found in {raw_dir}")
        return

    for img_path in tqdm(images, desc=f"  {split}", unit="img"):
        stem         = img_path.stem
        gray_out     = gray_dir     / f"{stem}_gray.png"
        dithered_out = dithered_dir / f"{stem}_dithered.png"

        if gray_out.exists() and dithered_out.exists():
            continue

        img = Image.open(img_path).convert("L")
        if SCALE != 1:
            img = img.resize((int(img.width * SCALE), int(img.height * SCALE)), Image.LANCZOS)

        img.save(gray_out)
        _dither(img).save(dithered_out)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Warm up Numba JIT on a tiny image before the real work
    print("Compiling dither kernel …")
    _dither(Image.new("L", (4, 4)))
    print("  done\n")

    for split, url in _SPLITS.items():
        print(f"── {split} ──────────────────────────────────────")
        raw_dir = DATA_DIR / f"DIV2K_{split}_HR"
        download_and_extract(split, url, raw_dir)
        process_split(split, raw_dir)
        print()

    print("Dataset ready.")
    print(f"  grayscale : {DATA_DIR}/gray/{{train,valid}}/")
    print(f"  dithered  : {DATA_DIR}/dithered/{{train,valid}}/")


if __name__ == "__main__":
    main()

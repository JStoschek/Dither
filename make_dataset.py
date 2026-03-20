"""
Download DIV2K high-resolution images and produce a paired
grayscale / Floyd-Steinberg dithered dataset for ML training.

Directory layout after running:
  data/
    raw/train/       DIV2K_train_HR images (2K PNGs)
    raw/valid/       DIV2K_valid_HR images
    gray/train/      grayscale versions at SCALE
    gray/valid/
    dithered/train/  Floyd-Steinberg dithered versions at SCALE
    dithered/valid/
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from PIL import Image

import floyd_steinberg

# ── config ────────────────────────────────────────────────────────────────────
SCALE    = 1   # resize factor applied before dithering (full-res 2K is slow)
DATA_DIR = Path("data")

_SPLITS = {
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}
# ──────────────────────────────────────────────────────────────────────────────


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb  = downloaded / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} MB", end="", flush=True)


def download_and_extract(split: str, url: str, raw_dir: Path):
    """Download zip for one split and extract into raw_dir."""
    zip_path = DATA_DIR / f"DIV2K_{split}_HR.zip"

    if not zip_path.exists():
        print(f"Downloading {split} HR images …")
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        print()  # newline after progress
    else:
        print(f"  zip already exists, skipping download: {zip_path}")

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        print(f"Extracting to {raw_dir} …")
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith((".png", ".jpg"))]
            for i, member in enumerate(members, 1):
                zf.extract(member, DATA_DIR)
                print(f"\r  {i}/{len(members)}", end="", flush=True)
        print()
    else:
        print(f"  already extracted, skipping: {raw_dir}")


def process_split(split: str, raw_dir: Path):
    """Convert every image in raw_dir to grayscale + dithered, save to data/."""
    gray_dir     = DATA_DIR / "gray"     / split
    dithered_dir = DATA_DIR / "dithered" / split
    gray_dir.mkdir(parents=True, exist_ok=True)
    dithered_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(raw_dir.rglob("*.png")) + sorted(raw_dir.rglob("*.jpg"))
    total  = len(images)
    if total == 0:
        print(f"  no images found in {raw_dir}")
        return

    print(f"Processing {total} {split} images at scale {SCALE} …")
    for i, img_path in enumerate(images, 1):
        stem = img_path.stem
        gray_out     = gray_dir     / f"{stem}_gray.png"
        dithered_out = dithered_dir / f"{stem}_dithered.png"

        # skip if both outputs already exist (makes the script resumable)
        if gray_out.exists() and dithered_out.exists():
            print(f"\r  {i}/{total} (cached)", end="", flush=True)
            continue

        img  = Image.open(img_path).convert("L")
        if SCALE != 1.0:
            new_size = (int(img.width * SCALE), int(img.height * SCALE))
            img = img.resize(new_size, Image.LANCZOS)

        # save grayscale
        img.save(gray_out)

        # dither
        pixels = [float(p) for p in img.getdata()]
        result = floyd_steinberg.dither(pixels, img.width, img.height)
        dithered_img = Image.new("L", (img.width, img.height))
        dithered_img.putdata([int(max(0, min(255, p))) for p in result])
        dithered_img.save(dithered_out)

        print(f"\r  {i}/{total}", end="", flush=True)

    print(f"\r  {total}/{total} done          ")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    for split, url in _SPLITS.items():
        print(f"\n── {split} ──────────────────────────────────────")
        # DIV2K extracts into a folder named DIV2K_<split>_HR
        raw_dir = DATA_DIR / f"DIV2K_{split}_HR"
        download_and_extract(split, url, raw_dir)
        process_split(split, raw_dir)

    print("\nDataset ready.")
    print(f"  grayscale : {DATA_DIR}/gray/{{train,valid}}/")
    print(f"  dithered  : {DATA_DIR}/dithered/{{train,valid}}/")


if __name__ == "__main__":
    main()

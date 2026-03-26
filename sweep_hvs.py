"""
Sweep per-scale HVS loss weights: train a short run for each config,
produce a dithered sample from each, and stitch them into a comparison grid.

Usage:
    python sweep_hvs.py                          # default configs, 15 epochs
    python sweep_hvs.py --epochs 25 --image cat.jpg --scale 0.5

Output:
    output/sweep_hvs_grid.png — labelled grid of all results
    output/sweep_hvs/<run_name>/best.pt — checkpoint for each config
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DitherDataset
from dither_net import DitherNet
from losses import HVSLoss
import floyd_steinberg

# ── sweep configs ─────────────────────────────────────────────────────────────
# (name, [fine_weight, medium_weight, coarse_weight])
CONFIGS = [
    ("balanced",       [1, 1, 1]),
    ("fine_heavy",     [4, 1, 1]),
    ("medium_heavy",   [1, 4, 1]),
    ("coarse_heavy",   [1, 1, 4]),
    ("no_fine",        [0, 1, 1]),
    ("no_medium",      [1, 0, 1]),
    ("no_coarse",      [1, 1, 0]),
    ("fine_only",      [1, 0, 0]),
    ("medium_only",    [0, 1, 0]),
]

BATCH_SIZE = 16
CROP_SIZE  = 128
LR         = 1e-4


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one(scale_weights, n_epochs, train_loader, valid_loader, device):
    """Train a fresh model with the given scale weights, return best model state."""
    model     = DitherNet().to(device)
    hvs_loss  = HVSLoss(weights=scale_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val   = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        # ── train ──
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            noise = torch.rand_like(x)
            logits = model(x, noise)
            loss = hvs_loss(logits, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ── validate ──
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                noise = torch.rand_like(x)
                logits = model(x, noise)
                val_total += hvs_loss(logits, x).item()
        val_loss = val_total / len(valid_loader)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_val


def infer(model_state, img, device):
    """Run inference on a PIL image, return binary PIL image."""
    model = DitherNet().to(device)
    model.load_state_dict(model_state)
    model.eval()

    arr  = np.array(img, dtype=np.float32) / 255.0
    gray = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(gray)
    binary = (logits.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def make_grid(panels, ncols=5):
    """Stitch labelled panels into a grid image."""
    label_h = 22
    w, h = panels[0][1].size
    nrows = (len(panels) + ncols - 1) // ncols
    grid_w = w * ncols
    grid_h = (h + label_h) * nrows
    canvas = Image.new("L", (grid_w, grid_h), color=200)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font = ImageFont.load_default()

    for idx, (label, img) in enumerate(panels):
        row, col = divmod(idx, ncols)
        x0 = col * w
        y0 = row * (h + label_h)
        canvas.paste(img, (x0, y0))

        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x0 + (w - tw) // 2, y0 + h + 3), label, fill=0, font=font)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Sweep HVS scale weights")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per config (default: 15)")
    parser.add_argument("--image", default="cat.jpg", help="Test image for visual comparison")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize factor for test image")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Configs: {len(CONFIGS)}   Epochs/config: {args.epochs}\n")

    # ── data ──
    train_ds = DitherDataset(split="train", crop_size=CROP_SIZE,
                             random_crop=True, hflip=True, vflip=True, rotate=True)
    valid_ds = DitherDataset(split="valid", crop_size=CROP_SIZE,
                             random_crop=False, hflip=False, vflip=False, rotate=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=pin)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin)
    print(f"Train: {len(train_ds):,}   Valid: {len(valid_ds):,}\n")

    # ── test image ──
    test_img = Image.open(args.image).convert("L")
    if args.scale != 1.0:
        test_img = test_img.resize(
            (int(test_img.width * args.scale), int(test_img.height * args.scale)),
            Image.LANCZOS,
        )
    print(f"Test image: {args.image} ({test_img.width}×{test_img.height})\n")

    # ── reference panels ──
    panels = [("original", test_img)]

    # Floyd-Steinberg reference
    fs_pixels = [float(p) for p in test_img.getdata()]
    fs_result = floyd_steinberg.dither(fs_pixels, test_img.width, test_img.height)
    fs_img = Image.new("L", test_img.size)
    fs_img.putdata([int(max(0, min(255, p))) for p in fs_result])
    panels.append(("floyd-steinberg", fs_img))

    # ── sweep ──
    sweep_dir = Path("output/sweep_hvs")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    for name, weights in CONFIGS:
        label = f"{name}\n[{weights[0]},{weights[1]},{weights[2]}]"
        print(f"── {name}  weights={weights} ──")

        best_state, best_val = train_one(
            weights, args.epochs, train_loader, valid_loader, device,
        )
        print(f"  best val={best_val:.6f}")

        # save checkpoint
        run_dir = sweep_dir / name
        run_dir.mkdir(exist_ok=True)
        torch.save({"model_state": best_state, "weights": weights, "val_loss": best_val},
                    run_dir / "best.pt")

        # inference
        result_img = infer(best_state, test_img, device)
        result_img.save(run_dir / "sample.png")
        panels.append((label, result_img))

        print()

    # ── final grid ──
    grid = make_grid(panels)
    out_path = Path("output/sweep_hvs_grid.png")
    grid.save(out_path)
    print(f"Grid saved to {out_path}")


if __name__ == "__main__":
    main()

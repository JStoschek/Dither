"""
Run DitherNet on an image and display the result alongside Floyd-Steinberg.

Usage:
    python infer.py cat.jpg
    python infer.py cat.jpg --checkpoint checkpoints/best.pt --scale 0.5
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from dither_net import DitherNet
import floyd_steinberg


CKPT_DEFAULT = Path("checkpoints/best.pt")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: Path, device: torch.device) -> DitherNet:
    model = DitherNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def dither_model(model: DitherNet, img: Image.Image, device: torch.device) -> Image.Image:
    arr = np.array(img, dtype=np.float32) / 255.0
    gray = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # 1×1×H×W
    with torch.no_grad():
        logits = model(gray)  # noise=None → model generates fresh noise
    binary = (logits.squeeze() > 0).cpu().numpy().astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def dither_fs(img: Image.Image) -> Image.Image:
    pixels = [float(p) for p in img.getdata()]
    result = floyd_steinberg.dither(pixels, img.width, img.height)
    out = Image.new("L", img.size)
    out.putdata([int(max(0, min(255, p))) for p in result])
    return out


def make_comparison(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    label_h = 22
    w, h = panels[0][1].size
    total_w = w * len(panels)
    canvas = Image.new("L", (total_w, h + label_h), color=200)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()

    for i, (label, img) in enumerate(panels):
        canvas.paste(img, (i * w, 0))
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((i * w + (w - text_w) // 2, h + 3), label, fill=0, font=font)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--checkpoint", default=str(CKPT_DEFAULT), help="Model checkpoint (.pt)")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize factor before dithering")
    parser.add_argument("--save", metavar="PATH", help="Save comparison image instead of displaying")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading checkpoint: {ckpt}")

    model = load_model(ckpt, device)

    img = Image.open(args.image).convert("L")
    if args.scale != 1.0:
        img = img.resize((int(img.width * args.scale), int(img.height * args.scale)), Image.LANCZOS)
    print(f"Image: {img.width}×{img.height}")

    print("Running Floyd-Steinberg …")
    fs_out = dither_fs(img)

    print("Running DitherNet …")
    ml_out = dither_model(model, img, device)

    comparison = make_comparison([
        ("original (gray)", img),
        ("floyd-steinberg",  fs_out),
        ("DitherNet",        ml_out),
    ])

    if args.save:
        comparison.save(args.save)
        print(f"Saved {args.save}")
    else:
        comparison.show()


if __name__ == "__main__":
    main()

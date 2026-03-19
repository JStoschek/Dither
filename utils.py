from PIL import Image
import os


def load(path, scale=1.0):
    img = Image.open(path)
    base_name = os.path.splitext(path)[0]
    print(f"Opened: {path}")
    print(f"Size: {img.size[0]}x{img.size[1]}  Mode: {img.mode}")

    gray = img.convert("L")
    if scale != 1.0:
        new_size = (int(img.width * scale), int(img.height * scale))
        gray = gray.resize(new_size, Image.LANCZOS)
        print(f"Scaled to: {gray.size[0]}x{gray.size[1]}")

    width, height = gray.size
    pixels = [float(p) for p in gray.getdata()]
    return pixels, width, height, base_name


OUTPUT_DIR = "output"


def save(pixels, width, height, path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, os.path.basename(path))
    img = Image.new("L", (width, height))
    img.putdata([int(max(0, min(255, p))) for p in pixels])
    img.save(out)
    print(f"Saved {out}")

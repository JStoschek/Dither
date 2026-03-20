from PIL import Image, ImageDraw, ImageFont
import threshold, random_noise, blue_noise, floyd_steinberg, bayer, jarvis_judice_ninke
from utils import load

FILE = "cat.jpg"
SCALE = 0.25
LABEL_HEIGHT = 20

METHODS = [
    ("threshold",           lambda p, w, h: threshold.dither(p, w, h)),
    ("random noise",        lambda p, w, h: random_noise.dither(p, w, h)),
    ("blue noise",          lambda p, w, h: blue_noise.dither(p, w, h)),
    ("bayer 2x2",           lambda p, w, h: bayer.dither(p, w, h, size=2)),
    ("bayer 4x4",           lambda p, w, h: bayer.dither(p, w, h, size=4)),
    ("floyd-steinberg",     lambda p, w, h: floyd_steinberg.dither(p, w, h)),
    ("Jarvis Judice Ninke", lambda p, w, h: jarvis_judice_ninke.dither(p, w, h)),
]


def make_panel(pixels, width, height, label):
    """Dithered image with a label strip underneath."""
    panel = Image.new("L", (width, height + LABEL_HEIGHT), color=200)
    img = Image.new("L", (width, height))
    img.putdata([int(max(0, min(255, p))) for p in pixels])
    panel.paste(img, (0, 0))

    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    x = (width - text_w) // 2
    draw.text((x, height + 2), label, fill=0, font=font)
    return panel


if __name__ == "__main__":
    pixels, width, height, base_name = load(FILE, SCALE)

    panels = []
    for label, fn in METHODS:
        print(f"Running {label}...")
        result = fn(pixels, width, height)
        panels.append(make_panel(result, width, height, label))

    total_width = width * len(panels)
    total_height = height + LABEL_HEIGHT
    comparison = Image.new("L", (total_width, total_height), color=200)
    for i, panel in enumerate(panels):
        comparison.paste(panel, (i * width, 0))

    out = f"output/{base_name}_compare_{SCALE}.png"
    comparison.save(out)
    print(f"Saved {out}")

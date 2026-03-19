import random
from utils import load, save

FILE = "cat.jpg"
SCALE = 0.25
STRENGTH = .75  # 0.0 = hard threshold, 1.0 = full noise


def dither(pixels, width, height, strength=STRENGTH):
    return [
        255.0 if p > 128 + (random.uniform(0, 255) - 128) * strength else 0.0
        for p in pixels
    ]


if __name__ == "__main__":
    pixels, width, height, base_name = load(FILE, SCALE)
    result = dither(pixels, width, height)
    save(result, width, height, f"{base_name}_random_noise_{SCALE}.png")

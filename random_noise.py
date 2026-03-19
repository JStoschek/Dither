import random
from utils import load, save

FILE = "cat.jpg"
SCALE = 0.25
NOISE = 64


def dither(pixels, width, height):
    return [
        255.0 if p + random.uniform(-NOISE, NOISE) >= 128 else 0.0
        for p in pixels
    ]


if __name__ == "__main__":
    pixels, width, height, base_name = load(FILE, SCALE)
    result = dither(pixels, width, height)
    save(result, width, height, f"{base_name}_random_noise_{SCALE}.png")

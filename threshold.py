from utils import load, save

FILE = "cat.jpg"
SCALE = 0.25
THRESHOLD = 128


def dither(pixels, width, height):
    return [255.0 if p >= THRESHOLD else 0.0 for p in pixels]


if __name__ == "__main__":
    pixels, width, height, base_name = load(FILE, SCALE)
    result = dither(pixels, width, height)
    save(result, width, height, f"{base_name}_threshold_{SCALE}.png")

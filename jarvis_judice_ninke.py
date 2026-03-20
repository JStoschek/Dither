from utils import load, save

FILE = "cat.jpg"
SCALE = 0.25

_DIFFUSION = [
    (0,  1,  7), (0,  2,  5), (1, -2,  3), (1, -1,  5), (1,  0,  7),
    (1,  1,  5), (1,  2,  3), (2, -2,  1), (2, -1,  3), (2,  0,  5),
    (2,  1,  3), (2,  2,  1),
]


def dither(pixels, width, height):
    buf = list(pixels)

    for y in range(height):
        for x in range(width):
            old = buf[y * width + x]
            new = 255.0 if old >= 128 else 0.0
            buf[y * width + x] = new
            err = old - new

            for dy, dx, weight in _DIFFUSION:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    buf[ny * width + nx] += err * weight / 48

    return buf


if __name__ == "__main__":
    pixels, width, height, base_name = load(FILE, SCALE)
    result = dither(pixels, width, height)
    save(result, width, height, f"{base_name}_jjn_{SCALE}.png")

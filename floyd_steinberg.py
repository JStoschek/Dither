from utils import load, save

FILE = "cat.jpg"
SCALE = 0.03


def dither(pixels, width, height):
    buf = list(pixels)

    for y in range(height):
        for x in range(width):
            old = buf[y * width + x]
            new = 255.0 if old >= 128 else 0.0
            buf[y * width + x] = new
            err = old - new

            if x + 1 < width:
                buf[y * width + (x + 1)] += err * 7 / 16
            if y + 1 < height:
                if x - 1 >= 0:
                    buf[(y + 1) * width + (x - 1)] += err * 3 / 16
                buf[(y + 1) * width + x] += err * 5 / 16
                if x + 1 < width:
                    buf[(y + 1) * width + (x + 1)] += err * 1 / 16

    return buf


if __name__ == "__main__":
    pixels, width, height, base_name = load(FILE, SCALE)
    result = dither(pixels, width, height)
    save(result, width, height, f"{base_name}_floyd_steinberg_{SCALE}.png")

import numpy as np
from utils import load, save

FILE = "cat.jpg"
SCALE = 0.25
TILE = 64       # blue noise texture tile size
STRENGTH = .75  # 0.0 = hard threshold, 1.0 = full noise


def make_blue_noise(size=64, seed=42):
    """Generate a blue noise tile via frequency-domain filtering of white noise."""
    rng = np.random.default_rng(seed)
    white = rng.random((size, size))

    # Build a high-pass frequency weight: weight each frequency by its magnitude
    freq = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freq, freq)
    magnitude = np.sqrt(fx**2 + fy**2)
    magnitude[0, 0] = 1  # avoid divide-by-zero at DC; DC will be zeroed anyway

    # Filter: amplify high frequencies, suppress low
    F = np.fft.fft2(white)
    F_filtered = F * magnitude
    filtered = np.real(np.fft.ifft2(F_filtered))

    # Rank-order normalize to [0, 255] so thresholds are uniformly spread
    flat = filtered.ravel()
    order = np.argsort(flat)
    result = np.empty_like(flat)
    result[order] = np.linspace(0, 255, len(flat))
    return result.reshape(size, size)


_BLUE_NOISE = make_blue_noise(TILE)


def dither(pixels, width, height, strength=STRENGTH):
    result = []
    for i, p in enumerate(pixels):
        x = i % width
        y = i // width
        threshold = 128 + (_BLUE_NOISE[y % TILE, x % TILE] - 128) * strength
        result.append(255.0 if p > threshold else 0.0)
    return result


if __name__ == "__main__":
    pixels, width, height, base_name = load(FILE, SCALE)
    result = dither(pixels, width, height)
    save(result, width, height, f"{base_name}_blue_noise_{SCALE}.png")

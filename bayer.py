from utils import load, save

FILE = "cat.jpg"
SCALE = 0.25
SIZE = 4        # 2 for 2x2 matrix, 4 for 4x4 matrix
STRENGTH = 1.0  # 0.0 = hard threshold, 1.0 = full range

BAYER_2X2 = [
    [0, 2],
    [3, 1],
]

BAYER_4X4 = [
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
]


def make_threshold_matrix(bayer):
    """Normalize a Bayer matrix to [0, 255] thresholds."""
    n = len(bayer) * len(bayer[0])
    return [[(v + 0.5) / n * 255 for v in row] for row in bayer]


_MATRICES = {
    2: make_threshold_matrix(BAYER_2X2),
    4: make_threshold_matrix(BAYER_4X4),
}


def dither(pixels, width, height, size=SIZE, strength=STRENGTH):
    matrix = _MATRICES[size]
    result = []
    for i, p in enumerate(pixels):
        x = i % width
        y = i // width
        threshold = 128 + (matrix[y % size][x % size] - 128) * strength
        result.append(255.0 if p > threshold else 0.0)
    return result


if __name__ == "__main__":
    pixels, width, height, base_name = load(FILE, SCALE)
    result = dither(pixels, width, height)
    save(result, width, height, f"{base_name}_bayer{SIZE}x{SIZE}_{SCALE}.png")

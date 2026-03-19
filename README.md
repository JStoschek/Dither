# Dither

A collection of image dithering algorithms implemented in Python.

## Scripts

| Script | Method | Key constants |
|---|---|---|
| `threshold.py` | Hard threshold at 128 | `THRESHOLD`, `SCALE` |
| `random_noise.py` | Uniform random noise threshold | `STRENGTH`, `SCALE` |
| `blue_noise.py` | 64×64 blue noise tile (FFT-generated) | `STRENGTH`, `SCALE` |
| `bayer.py` | Ordered dithering (Bayer matrix) | `SIZE` (2 or 4), `STRENGTH`, `SCALE` |
| `floyd_steinberg.py` | Error diffusion | `SCALE` |
| `compare.py` | Runs all methods, saves side-by-side comparison | `SCALE` |

`STRENGTH` controls noise intensity: `0.0` = hard threshold, `1.0` = full range.

## Usage

Run any script directly:
```
python threshold.py
python floyd_steinberg.py
python compare.py
```

## Output

Each script saves to `<name>_<method>_<scale>.png`.
`compare.py` saves `<name>_compare_<scale>.png` with all methods side-by-side.

## Requirements

- Python 3
- Pillow
- numpy

import numpy as np
import torch
import torch.nn.functional as F

# Viewing parameters — adjust to match your display setup
KERNEL_SIZE = 32         # spatial kernel size (pixels)
VIEWING_DISTANCE_CM = 50 # observer distance from screen
DPI = 96                 # screen resolution


def make_csf_kernel(size=KERNEL_SIZE, viewing_distance_cm=VIEWING_DISTANCE_CM, dpi=DPI):
    """
    Build a spatial HVS kernel from the Mannos-Sakrison (1974) CSF:
        CSF(f) = 2.6 * (0.0192 + 0.114*f) * exp(-(0.114*f)^1.1)
    where f is in cycles/degree.

    Steps:
      1. Build a 2D frequency grid in cycles/degree.
      2. Evaluate CSF → frequency-domain filter.
      3. IFFT back to spatial domain, shift so the peak is centred.
      4. Normalise so sum(|kernel|) == 1.
    """
    # pixels per degree at the given viewing distance and DPI
    px_per_cm = dpi / 2.54
    ppd = px_per_cm * viewing_distance_cm * np.tan(np.radians(1))

    # frequency grid in cycles/pixel, then convert to cycles/degree
    freq = np.fft.fftfreq(size)          # cycles/pixel, in [-0.5, 0.5)
    fx, fy = np.meshgrid(freq, freq)
    f_cpd = np.sqrt(fx**2 + fy**2) * ppd  # cycles/degree

    # Mannos-Sakrison CSF
    csf = 2.6 * (0.0192 + 0.114 * f_cpd) * np.exp(-(0.114 * f_cpd) ** 1.1)

    # IFFT → spatial kernel, shift so DC is at the centre
    kernel = np.real(np.fft.ifft2(np.fft.ifftshift(csf)))
    kernel = np.fft.fftshift(kernel)

    # Normalise
    kernel = kernel / np.sum(np.abs(kernel))
    return kernel.astype(np.float32)


def convolve(image_np: np.ndarray, kernel_np: np.ndarray) -> np.ndarray:
    """
    Convolve a 2D float image with a 2D kernel using PyTorch.

    Args:
        image_np:  H×W numpy array, any float dtype
        kernel_np: K×K numpy array

    Returns:
        H×W numpy float32 array (same spatial size, reflect-padded)
    """
    H, W = image_np.shape
    K = kernel_np.shape[0]
    pad = K // 2

    img_t = torch.from_numpy(image_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    ker_t = torch.from_numpy(kernel_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # 1,1,K,K

    # Reflect padding keeps edge behaviour consistent with the infinite-tiling assumption
    img_padded = F.pad(img_t, (pad, pad, pad, pad), mode="reflect")
    result = F.conv2d(img_padded, ker_t)

    return result.squeeze().numpy()


# Precomputed default kernel — import this in dbs.py and future ml_dither.py
_KERNEL = make_csf_kernel()


if __name__ == "__main__":
    print(f"Kernel shape : {_KERNEL.shape}")
    print(f"sum(|kernel|): {np.sum(np.abs(_KERNEL)):.6f}")
    print(f"Kernel min   : {_KERNEL.min():.6f}")
    print(f"Kernel max   : {_KERNEL.max():.6f}")

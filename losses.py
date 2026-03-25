import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gaussian(sigma: float, size: int) -> np.ndarray:
    k = size // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return (g / g.sum()).astype(np.float32)


def _ste_binarize(soft: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator: forward pass returns hard binary {0, 1},
    backward pass passes gradients through as if no thresholding happened.
    """
    binary = (soft > 0.5).float()
    return binary + (soft - soft.detach())


# ── Multi-scale Gaussian HVS filter bank ─────────────────────────────────────
#
# sigma=0.5, size=3  → fine:   enforces correct local dot density
# sigma=1.5, size=9  → medium: enforces texture quality (dithering grain)
# sigma=4.0, size=25 → coarse: enforces large-area tone accuracy
#
_SCALES = [
    (0.5,  3),
    (1.5,  9),
    (4.0, 25),
]


class HVSLoss(nn.Module):
    """
    Multi-scale perceptual dithering loss.

    Binarises the prediction with a straight-through estimator, then blurs
    both the binary output and the continuous gray input with Gaussians at
    multiple spatial scales.  MSE is computed at each scale and summed.

    Fine scale  (σ=0.5): every dot must contribute correct local brightness.
    Medium scale (σ=1.5): dot *patterns* must look uniform, not clumpy.
    Coarse scale (σ=4.0): large-area tone must match the original image.

    Together these produce output quality comparable to Floyd-Steinberg
    without ever needing to match an FS target pixel-for-pixel.
    """

    def __init__(self, scales: list[tuple[float, int]] | None = None):
        super().__init__()
        if scales is None:
            scales = _SCALES

        kernels = []
        pads = []
        for sigma, size in scales:
            k = _make_gaussian(sigma, size)
            kernels.append(torch.from_numpy(k).unsqueeze(0).unsqueeze(0))
            pads.append(size // 2)

        # Store as buffers so they move to the right device with model.to(device)
        for i, k in enumerate(kernels):
            self.register_buffer(f"kernel_{i}", k)
        self.pads = pads
        self.n_scales = len(scales)

    def _filter(self, x: torch.Tensor, scale_idx: int) -> torch.Tensor:
        kernel = getattr(self, f"kernel_{scale_idx}")
        pad = self.pads[scale_idx]
        return F.conv2d(F.pad(x, (pad,) * 4, mode="reflect"), kernel)

    def forward(self, logits: torch.Tensor, gray: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) — raw model output
            gray:   (B, 1, H, W) — grayscale input in [0, 1]
        """
        binary = _ste_binarize(torch.sigmoid(logits))

        loss = torch.tensor(0.0, device=logits.device)
        for i in range(self.n_scales):
            loss = loss + F.mse_loss(self._filter(binary, i), self._filter(gray, i))

        return loss

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


# ── Default multi-scale config ───────────────────────────────────────────────
#
#   (sigma, kernel_size, weight)
#
# Fine   (σ=0.5):  correct local dot density
# Medium (σ=1.5):  dithering texture / grain quality
# Coarse (σ=4.0):  large-area tone accuracy
#
DEFAULT_SCALES = [
    (0.5,  3),
    (1.5,  9),
    (4.0, 25),
]
DEFAULT_WEIGHTS = [1.0, 1.0, 1.0]


class HVSLoss(nn.Module):
    """
    Multi-scale perceptual dithering loss with per-scale weights.

    Binarises the prediction with a straight-through estimator, then blurs
    both the binary output and the continuous gray input with Gaussians at
    multiple spatial scales.  Weighted MSE is computed at each scale.

    Args:
        scales:  list of (sigma, kernel_size) tuples
        weights: list of per-scale loss multipliers, same length as scales.
                 [1,1,1] = balanced, [4,1,1] = emphasize fine dots,
                 [1,1,4] = emphasize smooth tone, etc.
    """

    def __init__(
        self,
        scales: list[tuple[float, int]] | None = None,
        weights: list[float] | None = None,
    ):
        super().__init__()
        if scales is None:
            scales = DEFAULT_SCALES
        if weights is None:
            weights = DEFAULT_WEIGHTS

        assert len(weights) == len(scales), \
            f"weights length ({len(weights)}) must match scales length ({len(scales)})"

        self.scale_weights = weights
        pads = []
        for i, (sigma, size) in enumerate(scales):
            k = _make_gaussian(sigma, size)
            self.register_buffer(f"kernel_{i}", torch.from_numpy(k).unsqueeze(0).unsqueeze(0))
            pads.append(size // 2)

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
            w = self.scale_weights[i]
            if w > 0:
                loss = loss + w * F.mse_loss(self._filter(binary, i), self._filter(gray, i))

        return loss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_hvs_gaussian(sigma: float = 1.5, size: int = 9) -> np.ndarray:
    """
    Gaussian approximation of the HVS lowpass blur, normalised to sum = 1
    (unit DC gain).  sigma = 1.5 px puts the −3 dB point at ≈ 0.11 c/px
    (≈ 3.5 cpd at 33 ppd), roughly where the CSF rolls off.
    """
    k = size // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    return (g / g.sum()).astype(np.float32)


def _ste_binarize(soft: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator: forward pass returns hard binary {0, 1},
    backward pass passes gradients through as if no thresholding happened.

    Without this, the model can cheat the HVS loss by outputting continuous
    values ≈ gray (blur(soft) ≈ blur(gray) → loss ≈ 0), which thresholds
    to hard threshold dithering.  With STE the loss evaluates the actual
    binary pattern, so the model must learn to arrange 0s and 1s such that
    blur(binary) ≈ blur(gray) — which IS dithering.
    """
    binary = (soft > 0.5).float()
    return binary + (soft - soft.detach())  # forward: binary, backward: ∂soft


class HVSLoss(nn.Module):
    """
    Perceptual dithering loss: MSE between the HVS-blurred **binary**
    prediction and the HVS-blurred continuous grayscale input.

    Uses a straight-through estimator so the output is truly binary in
    the forward pass (the loss sees 0s and 1s, not soft probabilities)
    while gradients still flow through sigmoid for backprop.
    """

    def __init__(self, sigma: float = 1.5, kernel_size: int = 9):
        super().__init__()
        kernel_np = _make_hvs_gaussian(sigma, kernel_size)
        self.pad = kernel_size // 2
        self.register_buffer(
            "kernel",
            torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0),  # (1,1,K,K)
        )

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (self.pad,) * 4, mode="reflect"), self.kernel)

    def forward(self, logits: torch.Tensor, gray: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) — raw model output
            gray:   (B, 1, H, W) — grayscale input in [0, 1]
        """
        binary = _ste_binarize(torch.sigmoid(logits))
        return F.mse_loss(self._filter(binary), self._filter(gray))

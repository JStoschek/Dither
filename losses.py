import torch
import torch.nn as nn
import torch.nn.functional as F

from hvs import make_csf_kernel


class HVSLoss(nn.Module):
    """
    Perceptual loss in the HVS-filtered domain.

    Applies the CSF-derived spatial kernel to both the binary prediction
    (sigmoid of logits) and the continuous grayscale input, then returns
    MSE between the two filtered images.

    This gives the model a gradient signal that is spatial rather than
    pixel-wise: it is penalised for getting the local average brightness
    wrong in the frequency band the eye is most sensitive to, which is
    exactly the quality criterion DBS optimises analytically.

    Uses an odd kernel size (33) so reflect-padding gives an output that
    is exactly the same spatial size as the input.
    """

    def __init__(self, kernel_size: int = 33):
        super().__init__()
        kernel_np = make_csf_kernel(size=kernel_size)
        self.pad = kernel_size // 2
        # register_buffer: moves to the right device automatically with model.to(device)
        self.register_buffer(
            "kernel",
            torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0),  # (1,1,K,K)
        )

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply HVS kernel to a (B, 1, H, W) tensor."""
        return F.conv2d(F.pad(x, (self.pad,) * 4, mode="reflect"), self.kernel)

    def forward(self, logits: torch.Tensor, gray: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) — raw model output
            gray:   (B, 1, H, W) — grayscale input in [0, 1]
        """
        return F.mse_loss(self._filter(torch.sigmoid(logits)), self._filter(gray))

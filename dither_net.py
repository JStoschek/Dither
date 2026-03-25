import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class DitherNet(nn.Module):
    """
    Input: (B, 2, H, W) — channel 0 = grayscale [0,1], channel 1 = noise [0,1]

    The noise channel gives the model a per-pixel random seed so it can break
    the symmetry in mid-tones.  Without it, BCE training collapses to hard
    threshold because the model can only predict the marginal probability at
    each pixel (≈ 0.5 for gray ≈ 128), which thresholds to a flat field.
    """

    def __init__(self, in_channels: int = 2, hidden_channels: int = 64, num_blocks: int = 8):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )

        self.tail = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, gray: torch.Tensor, noise: torch.Tensor | None = None):
        """
        Args:
            gray:  (B, 1, H, W) grayscale input
            noise: (B, 1, H, W) uniform noise in [0, 1].
                   If None, generates fresh noise (for inference).
        """
        if noise is None:
            noise = torch.rand_like(gray)
        x = torch.cat([gray, noise], dim=1)  # (B, 2, H, W)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)   # logits
        return x

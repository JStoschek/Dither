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
    def __init__(self, in_channels: int = 1, hidden_channels: int = 64, num_blocks: int = 8):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )

        self.tail = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)   # logits
        return x

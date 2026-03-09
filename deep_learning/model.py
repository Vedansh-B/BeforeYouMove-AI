"""Simple CNN for chess position value regression."""

from __future__ import annotations

import torch
from torch import nn


class ChessValueCNN(nn.Module):
    """Small convolutional network mapping (12, 8, 8) -> scalar value."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = ChessValueCNN()
    dummy = torch.randn(4, 12, 8, 8)
    out = model(dummy)
    print("Input shape:", tuple(dummy.shape))
    print("Output shape:", tuple(out.shape))

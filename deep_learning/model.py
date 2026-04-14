"""Simple CNN for chess position value regression."""

from __future__ import annotations

import torch
from torch import nn


class ChessValueCNN(nn.Module):
    """Hybrid CNN + MLP model for board planes and scalar chess features."""

    def __init__(self, input_channels: int = 18, feature_dim: int = 24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.board_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
        )
        self.feature_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        board_x = self.features(x)
        board_x = self.board_head(board_x)
        feature_x = self.feature_head(scalar_features)
        combined = torch.cat((board_x, feature_x), dim=1)
        return self.head(combined)


if __name__ == "__main__":
    model = ChessValueCNN()
    dummy = torch.randn(4, 18, 8, 8)
    dummy_features = torch.randn(4, 24)
    out = model(dummy, dummy_features)
    print("Input shape:", tuple(dummy.shape))
    print("Feature shape:", tuple(dummy_features.shape))
    print("Output shape:", tuple(out.shape))

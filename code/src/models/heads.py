"""Cabezas de regresión comunes a Teacher y Students."""
from __future__ import annotations

import torch.nn as nn


class RegressionHead(nn.Module):
    """Recibe features 3D (B, C, T, H, W) o 2D (B, C, H, W); pool global → MLP."""

    def __init__(self, in_features: int, dropout: float = 0.5, out_features: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, out_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 4:  # [B, C, H, W] → añadir dim T
            x = x.unsqueeze(2)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x).squeeze(-1)

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


class DistributionHead(nn.Module):
    """Cabeza que produce logits sobre N bins (Score Distribution Learning).

    Drop-in replacement de RegressionHead para la fase 8 (MUSDL).

    Returns:
        [B, num_bins] logits (sin softmax). La pérdida MUSDL aplica log_softmax.
    """

    def __init__(self,
                 in_features: int,
                 num_bins: int,
                 dropout: float = 0.5,
                 hidden: int = 0):
        super().__init__()
        self.num_bins = num_bins
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        layers = [nn.Dropout(dropout)]
        if hidden > 0:
            layers += [nn.Linear(in_features, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            layers.append(nn.Linear(hidden, num_bins))
        else:
            layers.append(nn.Linear(in_features, num_bins))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(2)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)  # [B, num_bins]

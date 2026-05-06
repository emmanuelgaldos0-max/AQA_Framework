"""Wrapper que convierte un modelo Student de regresión en un modelo MUSDL.

Idea: tomar el backbone existente (MobileNetV3-Video o TSM-MobileNetV2) y
reemplazar su `head` (RegressionHead) por una `DistributionHead`. El `forward`
del backbone interno se preserva (mantiene mid_feat / final_feat para
compatibilidad con KD si se quisiera combinar).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .heads import DistributionHead


class MUSDLWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, num_bins: int, dropout: float = 0.5):
        super().__init__()
        self.base = base_model
        # Detectar dimensión del feature final del backbone
        if hasattr(base_model, "final_feat_channels"):
            in_features = int(base_model.final_feat_channels)
        else:
            raise AttributeError(
                "MUSDLWrapper requiere que el modelo base exponga `final_feat_channels`"
            )
        self.num_bins = num_bins
        self.head = DistributionHead(
            in_features=in_features, num_bins=num_bins, dropout=dropout
        )
        # Sustituye la cabeza del modelo base por una identidad: queremos
        # acceder a `final_feat` que el backbone guarda internamente.
        self.base.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ejecuta el backbone (que internamente actualiza self.base.final_feat).
        _ = self.base(x)
        feat = self.base.final_feat  # [B, C, T, H, W]
        return self.head(feat)  # [B, num_bins]

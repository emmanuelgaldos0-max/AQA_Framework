"""TSM-MobileNetV2: MobileNetV2 2D con Temporal Shift Module inyectado en los
bloques invertidos residuales. Ligero para dispositivos con recursos limitados.

Estrategia:
  - Tomar `torchvision.models.mobilenet_v2` pre-entrenado en ImageNet.
  - Reshape entrada [B, C, T, H, W] → [B*T, C, H, W].
  - Envolver cada InvertedResidual con `TemporalShift(block, n_segment=T, n_div=8)`.
  - Pool global espacio-temporal → cabeza de regresión.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.mobilenetv2 import InvertedResidual

from .heads import RegressionHead
from .tsm import TemporalShift

# canales a la salida de bloques clave de MobileNetV2
# features[0..18]; vamos a usar features[7] como feature intermedia (64 canales)
# y features[-1] (1280 canales) como feature final.
_MID_BLOCK_IDX = 7
_MID_FEAT_CHANNELS = 64
_FINAL_FEAT_CHANNELS = 1280


class TSMMobileNetV2(nn.Module):
    def __init__(self,
                 clip_length: int,
                 pretrained: bool = True,
                 n_div: int = 8,
                 dropout: float = 0.5):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v2(weights=weights)
        features = list(base.features)

        # Envolver cada InvertedResidual con TemporalShift
        for i, layer in enumerate(features):
            if isinstance(layer, InvertedResidual):
                features[i] = TemporalShift(layer, n_segment=clip_length, n_div=n_div)

        self.features = nn.ModuleList(features)
        self.head = RegressionHead(in_features=_FINAL_FEAT_CHANNELS, dropout=dropout)
        self.clip_length = clip_length
        self.mid_feat: Optional[torch.Tensor] = None
        self.final_feat: Optional[torch.Tensor] = None

    @property
    def mid_feat_channels(self) -> int:
        return _MID_FEAT_CHANNELS

    @property
    def final_feat_channels(self) -> int:
        return _FINAL_FEAT_CHANNELS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        if T != self.clip_length:
            raise ValueError(f"clip_length={self.clip_length} ≠ T entrante={T}")
        x = x.transpose(1, 2).reshape(B * T, C, H, W)  # [B*T, C, H, W]

        mid_3d: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == _MID_BLOCK_IDX:
                # guardar versión 3D para KD
                mid_3d = x.view(B, T, *x.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()
        # x: [B*T, 1280, h, w]
        x3d = x.view(B, T, *x.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()  # [B, 1280, T, h, w]
        self.mid_feat = mid_3d
        self.final_feat = x3d
        return self.head(x3d)


def build_tsm_mobilenetv2(clip_length: int,
                          pretrained: bool = True,
                          n_div: int = 8,
                          dropout: float = 0.5) -> TSMMobileNetV2:
    return TSMMobileNetV2(
        clip_length=clip_length,
        pretrained=pretrained,
        n_div=n_div,
        dropout=dropout,
    )

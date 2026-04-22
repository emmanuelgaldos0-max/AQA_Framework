"""MobileNetV3-Large aplicado a video por procesado por frame + pool temporal.
No incluye módulos temporales explícitos, a propósito (§4.2.2 de la tesis):
mide el impacto de la destilación espacio-temporal sobre una arquitectura
puramente espacial.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .heads import RegressionHead

# índice del bloque intermedio y canales a la salida para KD
# mobilenet_v3_large.features tiene 17 bloques (0..16)
# features[6] salida ≈ 40 canales (en la versión large tras el 3er bneck)
# features[-1] salida = 960 canales
_MID_BLOCK_IDX = 6
_MID_FEAT_CHANNELS = 40
_FINAL_FEAT_CHANNELS = 960


class MobileNetV3Video(nn.Module):
    def __init__(self, clip_length: int, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v3_large(weights=weights)
        self.features = base.features  # Sequential
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
        x = x.transpose(1, 2).reshape(B * T, C, H, W)

        mid_3d: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == _MID_BLOCK_IDX:
                mid_3d = x.view(B, T, *x.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()
        final_3d = x.view(B, T, *x.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()
        self.mid_feat = mid_3d
        self.final_feat = final_3d
        return self.head(final_3d)


def build_mobilenetv3(clip_length: int, pretrained: bool = True, dropout: float = 0.5) -> MobileNetV3Video:
    return MobileNetV3Video(clip_length=clip_length, pretrained=pretrained, dropout=dropout)

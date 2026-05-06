"""MobileNetV2 plano para video, SIN Temporal Shift Module.

Este modelo es la ablación E7 del plan del asesor: igual base que
TSM-MobileNetV2 pero sin la inyección del módulo TSM. Procesa frame por
frame y agrega temporalmente sólo vía pool global. Sirve para aislar el
aporte del modelado temporal explícito.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from .heads import RegressionHead

# Mismas convenciones que tsm_mobilenetv2.py para que el wrapper de KD funcione.
_MID_BLOCK_IDX = 7
_MID_FEAT_CHANNELS = 64
_FINAL_FEAT_CHANNELS = 1280


class MobileNetV2Video(nn.Module):
    """MobileNetV2 aplicado frame-por-frame, agregación temporal por pool.

    No incluye TSM. Es el counterpart de TSM-MobileNetV2 con el componente
    temporal removido. Usado para la ablación E7 del plan del asesor.
    """

    def __init__(self,
                 clip_length: int,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        base = mobilenet_v2(weights=weights)
        self.features = base.features  # Sequential, sin TSM
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
        B, C, T, H, W = x.shape
        if T != self.clip_length:
            raise ValueError(f"clip_length={self.clip_length} ≠ T entrante={T}")
        x = x.transpose(1, 2).reshape(B * T, C, H, W)

        mid_3d: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == _MID_BLOCK_IDX:
                mid_3d = x.view(B, T, *x.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()
        x3d = x.view(B, T, *x.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()
        self.mid_feat = mid_3d
        self.final_feat = x3d
        return self.head(x3d)


def build_mobilenetv2_video(clip_length: int, pretrained: bool = True,
                            dropout: float = 0.5) -> MobileNetV2Video:
    return MobileNetV2Video(clip_length=clip_length, pretrained=pretrained, dropout=dropout)

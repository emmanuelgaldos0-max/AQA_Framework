"""I3D-R50 (pre-entrenado en Kinetics-400) adaptado a regresión AQA.

Estructura de `pytorchvideo.models.i3d_r50`:
    blocks[0]: ResNetBasicStem          (conv3d inicial)
    blocks[1]: ResStage                 stage1 → 256 canales
    blocks[2]: MaxPool3d
    blocks[3]: ResStage                 stage2 → 512 canales   ← hook para KD
    blocks[4]: ResStage                 stage3 → 1024 canales
    blocks[5]: ResStage                 stage4 → 2048 canales
    blocks[6]: ResNetBasicHead          (400 clases Kinetics)

Reemplazamos blocks[6] por RegressionHead y exponemos blocks[3] como
`feat_mid` para la destilación de atención y alineación temporal.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .heads import RegressionHead

# canales a la salida de cada ResStage del I3D-R50
_MID_FEAT_CHANNELS = 512      # blocks[3]
_FINAL_FEAT_CHANNELS = 2048   # blocks[5]


class I3DRegressor(nn.Module):
    """Teacher para AQA. Forward devuelve score en [0,1] y guarda features
    intermedias en `self.mid_feat` (última invocación) para KD."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        import pytorchvideo.models.hub as hub

        self.backbone = hub.i3d_r50(pretrained=pretrained)
        # remover la head original (blocks[-1])
        self.backbone.blocks = nn.ModuleList(list(self.backbone.blocks[:-1]))
        self.head = RegressionHead(in_features=_FINAL_FEAT_CHANNELS, dropout=dropout)
        self.mid_feat: Optional[torch.Tensor] = None

    @property
    def mid_feat_channels(self) -> int:
        return _MID_FEAT_CHANNELS

    @property
    def final_feat_channels(self) -> int:
        return _FINAL_FEAT_CHANNELS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i == 3:
                self.mid_feat = x  # [B, 512, T', H', W']
        # x: [B, 2048, T'', H'', W'']
        self.final_feat = x
        return self.head(x)


def build_i3d(pretrained: bool = True, dropout: float = 0.5) -> I3DRegressor:
    return I3DRegressor(pretrained=pretrained, dropout=dropout)

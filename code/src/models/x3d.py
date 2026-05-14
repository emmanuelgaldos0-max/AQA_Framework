"""X3D-M (Feichtenhofer 2020) adaptado a regresión AQA.

X3D-M es una arquitectura eficiente expandida progresivamente desde 2D-CNN
a 3D. Tiene ~3.79M parámetros, comparable en tamaño con MobileNetV3 pero
con convoluciones 3D explícitas.

Se usa como referencia adicional en el benchmark Pareto del Cap_5:
permite comparar Students 2D+TSM vs un Student 3D liviano.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .heads import RegressionHead

_FINAL_FEAT_CHANNELS = 192  # canales a la salida del stage final de X3D-M


class X3DRegressor(nn.Module):
    """X3D-M para AQA. Forward devuelve score en [0,1]."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.5,
                 variant: str = "x3d_m"):
        super().__init__()
        import pytorchvideo.models.hub as hub
        builder = getattr(hub, variant)
        self.backbone = hub.x3d_m(pretrained=pretrained) if variant == "x3d_m" \
            else builder(pretrained=pretrained)
        # remover head Kinetics
        self.backbone.blocks = nn.ModuleList(list(self.backbone.blocks[:-1]))
        self.head = RegressionHead(in_features=_FINAL_FEAT_CHANNELS, dropout=dropout)
        self.final_feat: Optional[torch.Tensor] = None

    @property
    def final_feat_channels(self) -> int:
        return _FINAL_FEAT_CHANNELS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.backbone.blocks:
            x = block(x)
        self.final_feat = x
        return self.head(x)


def build_x3d(pretrained: bool = True, dropout: float = 0.5,
              variant: str = "x3d_m") -> X3DRegressor:
    return X3DRegressor(pretrained=pretrained, dropout=dropout, variant=variant)

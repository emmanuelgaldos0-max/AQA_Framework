"""SlowFast-R50 (pre-entrenado en Kinetics-400) adaptado a regresión AQA.

Teacher adicional sugerido por el asesor (E1 extendido): además de I3D
(2017), reportar SlowFast (2019) como referencia de Teacher 3D moderno
en AQA-7.

Estructura de `pytorchvideo.models.hub.slowfast_r50`:
    blocks[0..4]: MultiPathWayWithFuse  (slow + fast pathways con fusión)
    blocks[5]:    PoolConcatPathway     (concat de los dos paths)
    blocks[6]:    ResNetBasicHead       (400 clases Kinetics)

El modelo recibe una lista [slow_input, fast_input] donde:
    slow_input: [B, C, T_slow, H, W]   T_slow = T_fast / alpha (alpha=4)
    fast_input: [B, C, T_fast, H, W]
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .heads import RegressionHead

# Canales a la salida del PoolConcatPathway (slow_chans + fast_chans)
_FINAL_FEAT_CHANNELS = 2048 + 256  # slow R50 → 2048, fast R50 → 256


class SlowFastRegressor(nn.Module):
    """Teacher SlowFast para AQA. Forward acepta `[B, C, T, H, W]` (igual que
    los Students) y construye internamente los dos paths con `alpha=4`.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.5,
                 alpha: int = 4):
        super().__init__()
        import pytorchvideo.models.hub as hub
        self.backbone = hub.slowfast_r50(pretrained=pretrained)
        # remover (a) head Kinetics y (b) PoolConcatPathway con kernel hardcoded.
        # Reemplazamos el PoolConcatPathway por un pool adaptativo en `forward`,
        # lo que permite cualquier longitud de clip (no sólo T_fast=32).
        self.backbone.blocks = nn.ModuleList(list(self.backbone.blocks[:-2]))
        self.head = RegressionHead(in_features=_FINAL_FEAT_CHANNELS, dropout=dropout)
        self.alpha = alpha
        self.final_feat: Optional[torch.Tensor] = None

    @property
    def final_feat_channels(self) -> int:
        return _FINAL_FEAT_CHANNELS

    def _build_pathways(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Construye [slow, fast] desde un tensor único [B, C, T, H, W]."""
        # fast = todos los frames; slow = subsampleado por alpha
        fast = x
        slow = x[:, :, ::self.alpha, :, :].contiguous()
        return [slow, fast]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]; backbone espera [slow, fast]
        pathways = self._build_pathways(x)
        for block in self.backbone.blocks:
            pathways = block(pathways)
        # `pathways` ahora es lista [slow_feat, fast_feat] de tensores
        # [B, C, T', H', W']. Pool adaptativo por path → concatenar.
        import torch.nn.functional as F
        pooled = [F.adaptive_avg_pool3d(p, 1) for p in pathways]
        feat = torch.cat(pooled, dim=1)  # [B, 2304, 1, 1, 1]
        self.final_feat = feat
        return self.head(feat)


def build_slowfast(pretrained: bool = True, dropout: float = 0.5,
                   alpha: int = 4) -> SlowFastRegressor:
    return SlowFastRegressor(pretrained=pretrained, dropout=dropout, alpha=alpha)

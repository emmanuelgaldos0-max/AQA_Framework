"""Temporal Shift Module según Lin et al., ICCV 2019.

Envuelve un bloque espacial 2D y desplaza una fracción `1/n_div` de sus canales
a lo largo de la dimensión temporal antes de aplicar el bloque.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalShift(nn.Module):
    def __init__(self, block: nn.Module, n_segment: int, n_div: int = 8):
        super().__init__()
        self.block = block
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, H, W]
        nt, c, h, w = x.shape
        if nt % self.n_segment != 0:
            raise ValueError(f"TSM: batch*T={nt} no es múltiplo de n_segment={self.n_segment}")
        b = nt // self.n_segment
        x = x.view(b, self.n_segment, c, h, w)
        fold = c // self.n_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]                  # shift left
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]             # no shift
        x = out.view(nt, c, h, w)
        return self.block(x)

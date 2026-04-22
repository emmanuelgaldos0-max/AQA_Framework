"""Alineación temporal de representaciones intermedias entre Student y Teacher.

Proyecta las features intermedias (distintos canales) a un espacio común, colapsa
el eje espacial por promedio y aplica MSE cuadro a cuadro. Ajusta la longitud
temporal por interpolación lineal si difiere.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlignLoss(nn.Module):
    def __init__(self, s_channels: int, t_channels: int, common: int = 256):
        super().__init__()
        self.proj_s = nn.Conv3d(s_channels, common, kernel_size=1)
        self.proj_t = nn.Conv3d(t_channels, common, kernel_size=1)

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        # feats: [B, C, T, H, W]
        h_s = self.proj_s(feat_s).mean(dim=[3, 4])  # [B, C', T_s]
        h_t = self.proj_t(feat_t).mean(dim=[3, 4])  # [B, C', T_t]

        if h_s.shape[-1] != h_t.shape[-1]:
            h_s = F.interpolate(h_s, size=h_t.shape[-1], mode="linear", align_corners=False)
        return F.mse_loss(h_s, h_t)

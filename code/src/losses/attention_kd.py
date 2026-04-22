"""Destilación de conocimiento sobre mapas de atención espacio-temporales.

Sigue la receta clásica de Zagoruyko & Komodakis (ICLR 2017):
    A(F) = sum(|F|^p, axis=channels)       con p=2 por defecto.

Aquí operamos sobre features 3D [B, C, T, H, W]; los mapas resultantes son
[B, T, H, W]. Se ajustan a resolución común por interpolación trilineal.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def attention_map(feat: torch.Tensor, p: int = 2) -> torch.Tensor:
    """feat [B, C, T, H, W] → [B, T, H, W]."""
    if feat.dim() != 5:
        raise ValueError(f"Se esperaba tensor 5D, se recibió {feat.shape}")
    return feat.abs().pow(p).mean(dim=1)


def attention_kd_loss(feat_s: torch.Tensor,
                      feat_t: torch.Tensor,
                      mode: str = "kl",
                      eps: float = 1e-8) -> torch.Tensor:
    a_s = attention_map(feat_s)
    a_t = attention_map(feat_t)

    if a_s.shape != a_t.shape:
        a_s = F.interpolate(
            a_s.unsqueeze(1),
            size=a_t.shape[1:],
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)

    B = a_s.size(0)
    a_s_flat = a_s.reshape(B, -1)
    a_t_flat = a_t.reshape(B, -1)

    if mode == "kl":
        p_s = F.log_softmax(a_s_flat, dim=1)
        p_t = F.softmax(a_t_flat, dim=1)
        return F.kl_div(p_s, p_t, reduction="batchmean")
    if mode == "mse":
        # normalizar L2 antes de comparar (práctica estándar en KD de atención)
        a_s_n = F.normalize(a_s_flat, p=2, dim=1)
        a_t_n = F.normalize(a_t_flat, p=2, dim=1)
        return (a_s_n - a_t_n).pow(2).mean()
    raise ValueError(f"mode inválido: {mode}")

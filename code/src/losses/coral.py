"""CORAL loss (Sun & Saenko 2016) — alineación de momentos de segundo orden
entre features `source` y `target` para domain adaptation.

Referencia:
  Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation Alignment for
  Deep Domain Adaptation. ECCV 2016 Workshops.
  https://arxiv.org/abs/1607.01719

Uso típico:
  feat_s = student(clip_source)   # [B_s, D]
  feat_t = student(clip_target)   # [B_t, D]
  L_coral = coral_loss(feat_s, feat_t)
  L_total = L_reg(feat_s, y_s) + lambda * L_coral
"""
from __future__ import annotations

import torch


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    CORAL: ‖C_s − C_t‖_F^2 / (4 d^2), donde C_s y C_t son las matrices de
    covarianza de los features source y target.

    Args:
        source: [B_s, D]
        target: [B_t, D]

    Returns:
        scalar loss
    """
    if source.dim() != 2 or target.dim() != 2:
        raise ValueError(f"source/target deben ser [B, D], got {source.shape}, {target.shape}")
    d = source.size(1)

    # Covarianza de un batch de features [B, D]: C = (X - X.mean(0))^T (X - X.mean(0)) / (B-1)
    def _cov(x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        b = max(x.size(0) - 1, 1)
        xm = x - x.mean(dim=0, keepdim=True)
        return (xm.t() @ xm) / b

    cs = _cov(source)
    ct = _cov(target)
    diff = cs - ct
    return (diff * diff).sum() / (4.0 * d * d)

"""Score Distribution Learning (USDL/MUSDL, Tang et al. CVPR 2020).

En lugar de regresar un escalar, el modelo predice una distribución sobre N bins
de score. La etiqueta real `y` se convierte en una gaussiana discreta centrada en
`y` con desviación `sigma`. La pérdida es KL(p_target || p_pred).

En inferencia, se calcula la esperanza E[score] = Σ_i bin_i · p_i para volver a
un escalar comparable con el baseline.

Referencia:
  Tang, Y., Ni, Z., Zhou, J., Zhang, D., Lu, J., Wu, Y., & Zhou, J. (2020).
  Uncertainty-aware Score Distribution Learning for Action Quality Assessment.
  CVPR 2020.  https://arxiv.org/abs/2006.07665
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def make_target_distribution(scores: torch.Tensor,
                             num_bins: int,
                             sigma: float,
                             score_min: float = 0.0,
                             score_max: float = 1.0) -> torch.Tensor:
    """
    Convierte un batch de scores escalares en distribuciones gaussianas discretas.

    Args:
        scores: [B] valores en [score_min, score_max]
        num_bins: número de bins de la distribución
        sigma: desviación estándar de la gaussiana en unidades de bin
        score_min: cota inferior del rango de score
        score_max: cota superior del rango de score

    Returns:
        [B, num_bins] distribución de probabilidad (suma 1 en dim 1)
    """
    device = scores.device
    bins = torch.linspace(score_min, score_max, num_bins, device=device)  # [N]
    width = (score_max - score_min) / max(num_bins - 1, 1)
    sigma_score = sigma * width  # convertir sigma de "unidades de bin" a "unidades de score"

    diff = bins.unsqueeze(0) - scores.unsqueeze(1)  # [B, N]
    log_prob = -0.5 * (diff / sigma_score) ** 2
    # estabilidad numérica: restar el máximo por fila
    log_prob = log_prob - log_prob.max(dim=1, keepdim=True).values
    prob = torch.exp(log_prob)
    prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return prob


def musdl_kl_loss(pred_logits: torch.Tensor,
                  target_scores: torch.Tensor,
                  num_bins: int,
                  sigma: float,
                  score_min: float = 0.0,
                  score_max: float = 1.0,
                  reduction: str = "batchmean") -> torch.Tensor:
    """
    Pérdida MUSDL: KL(target_dist || pred_dist).

    Args:
        pred_logits: [B, num_bins] logits sin softmax
        target_scores: [B] escalares en [score_min, score_max]
        num_bins, sigma, score_min, score_max: ver `make_target_distribution`
        reduction: 'batchmean' | 'mean' | 'sum'
    """
    log_pred = F.log_softmax(pred_logits, dim=1)
    target = make_target_distribution(
        target_scores, num_bins, sigma, score_min, score_max
    )
    # F.kl_div asume pred ya en log-space y target en prob-space
    return F.kl_div(log_pred, target, reduction=reduction)


def expected_score(pred_logits: torch.Tensor,
                   num_bins: int,
                   score_min: float = 0.0,
                   score_max: float = 1.0) -> torch.Tensor:
    """E[score] = Σ_i bin_i · p_i, para volver a escalar comparable con baseline."""
    bins = torch.linspace(score_min, score_max, num_bins, device=pred_logits.device)
    p = F.softmax(pred_logits, dim=1)
    return (p * bins.unsqueeze(0)).sum(dim=1)

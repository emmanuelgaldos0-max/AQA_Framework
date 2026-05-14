"""Test-Time Adaptation (TTA) evaluator para cross-domain AQA.

Implementa BN-recalibration: el modelo entrenado en `source` recibe un breve
forward sobre clips del `target` (sin gradiente) con BN en modo `train`, lo
que actualiza las estadísticas running_mean / running_var a las del dominio
target. Luego se hace inferencia normal.

Es la primera aplicación documentada de TTA a AQA (búsqueda agente
2026-05-14: ningún paper AQA 2024-2026 reporta TTA).

Referencias generales:
  - Schneider et al. 2020 "Improving robustness against common corruptions by
    covariate shift adaptation" (NeurIPS).
  - Wang et al. 2021 "Tent: Fully Test-Time Adaptation by Entropy Minimization"
    (ICLR).

Aquí se implementa la variante BN-only (Schneider 2020), que es la más
estable y compatible con regresión (no requiere entropía sobre clasificación).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import mae, plcc, srcc


def _set_bn_train(model: nn.Module):
    """Pone BatchNorm en train mode; resto del modelo en eval (no actualizar pesos)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
            # Permitir actualización de running stats
            m.track_running_stats = True


def _reset_bn_stats(model: nn.Module):
    """Reinicia las estadísticas running_mean / running_var (las del source)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()


@torch.no_grad()
def adapt_bn(model: nn.Module, adapt_loader: DataLoader, device: str = "cuda",
             amp: bool = True, n_passes: int = 1,
             reset_first: bool = True) -> None:
    """Recalibra las estadísticas BN del modelo con clips del target.

    Args:
        model: modelo entrenado en source.
        adapt_loader: DataLoader sobre split del target (puede ser train sin labels).
        n_passes: cuántas pasadas hacer sobre el adapt_loader.
        reset_first: si True, borra las running stats del source antes de
                     adaptar (recomendado para TTA puro). Si False, mezcla
                     stats source + target.
    """
    if reset_first:
        _reset_bn_stats(model)

    _set_bn_train(model)

    for pass_idx in range(n_passes):
        for batch in tqdm(adapt_loader, desc=f"TTA BN-adapt pass {pass_idx+1}/{n_passes}",
                          leave=False):
            clips = batch["clip"].to(device, non_blocking=True)
            with autocast(enabled=amp):
                _ = model(clips)

    # Después de adaptar, volver a eval para inferencia normal
    model.eval()


@torch.no_grad()
def evaluate_with_tta(model: nn.Module, adapt_loader: DataLoader,
                      eval_loader: DataLoader, device: str = "cuda",
                      amp: bool = True, score_scale: float = 100.0,
                      n_passes: int = 1, reset_first: bool = True) -> Dict[str, float]:
    """Pipeline completo: adapta BN + evalúa SRCC/PLCC/MAE sobre eval_loader."""
    # 1) Adaptar BN al target
    adapt_bn(model, adapt_loader, device=device, amp=amp,
             n_passes=n_passes, reset_first=reset_first)

    # 2) Inferencia normal
    model.eval()
    preds, gts = [], []
    for batch in tqdm(eval_loader, desc="TTA eval", leave=False):
        clips = batch["clip"].to(device, non_blocking=True)
        targets = batch["score"].to(device, non_blocking=True)
        with autocast(enabled=amp):
            p = model(clips)
        preds.append(p.detach().float().cpu())
        gts.append(targets.detach().float().cpu())
    pred = torch.cat(preds).view(-1)
    gt = torch.cat(gts).view(-1)
    return {
        "srcc": srcc(pred, gt),
        "plcc": plcc(pred, gt),
        "mae": mae(pred, gt, scale=score_scale),
        "n": int(pred.numel()),
    }

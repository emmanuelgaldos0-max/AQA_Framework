"""Trainer con domain adaptation vía CORAL.

Combina la pérdida de regresión sobre el dominio `source` (con etiquetas) con
una pérdida de alineación CORAL entre features de source y features de
`target` (sin etiquetas). El resultado son features del Student más
invariantes al dominio, lo que mejora la transferencia zero-shot
source → target.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.losses.coral import coral_loss
from src.utils.metrics import mae, plcc, srcc

from .trainer import Trainer


class CoralTrainer(Trainer):
    """Trainer con domain adaptation por CORAL.

    Se asume que `train_loader` provee `source` (con scores), y que se pasa
    además un `target_loader` (sin scores) en `__init__`.
    """

    def __init__(self, model, cfg, train_loader, val_loader, target_loader, run_dir):
        super().__init__(model, cfg, train_loader, val_loader, run_dir)
        self.target_loader = target_loader
        self.target_iter = iter(self.target_loader)
        coral_cfg = self.cfg.get("coral", {})
        self.lambda_coral = float(coral_cfg.get("lambda", 0.1))
        self.freeze_bn = bool(coral_cfg.get("freeze_bn", True))

    def _set_bn_eval(self):
        """Congela las estadísticas de BatchNorm del modelo (running mean/var).
        Crítico cuando batch_size=1 (CORAL hace 2 forwards y duplica VRAM, así
        que es necesario reducir batch). Sin esto, BN se degrada con batch=1
        y destruye las features pre-entrenadas en ImageNet (ver BITACORA E13).
        """
        import torch.nn as nn
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    def _next_target(self):
        try:
            return next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.target_loader)
            return next(self.target_iter)

    def _pool_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Pool global espacio-temporal: [B, C, T, H, W] → [B, C]."""
        return feat.mean(dim=(2, 3, 4))

    def compute_loss(self, clips: torch.Tensor, targets: torch.Tensor,
                     epoch: int) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        if self.freeze_bn:
            self._set_bn_eval()
        # forward source
        preds = self.model(clips)
        l_reg = F.mse_loss(preds.view(-1), targets.view(-1))
        feat_s = self._pool_features(self.model.final_feat) if hasattr(self.model, "final_feat") \
                 else self._pool_features(self.model.base.final_feat)

        # forward target (sin labels)
        target_batch = self._next_target()
        clips_t = target_batch["clip"].to(self.device, non_blocking=True)
        with autocast(enabled=self.cfg.get("amp", True)):
            _ = self.model(clips_t)
            feat_t = self._pool_features(self.model.final_feat) if hasattr(self.model, "final_feat") \
                     else self._pool_features(self.model.base.final_feat)

        l_coral = coral_loss(feat_s.float(), feat_t.float())
        loss = l_reg + self.lambda_coral * l_coral
        return loss, {
            "loss_total": loss.item(),
            "loss_reg": l_reg.item(),
            "loss_coral": l_coral.item(),
        }, preds

"""Trainer especializado para Score Distribution Learning (MUSDL).

Diferencias frente a Trainer base:
  * `compute_loss`: usa `musdl_kl_loss` en vez de MSE.
  * `validate`: convierte los logits del modelo a un escalar mediante
    expected_score antes de calcular SRCC/PLCC/MAE.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.losses.score_distribution import expected_score, musdl_kl_loss
from src.utils.metrics import mae, plcc, srcc

from .trainer import Trainer


class MUSDLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        musdl_cfg = self.cfg.get("musdl", {})
        self.num_bins = int(musdl_cfg.get("num_bins", 100))
        self.sigma = float(musdl_cfg.get("sigma", 5.0))
        self.score_min = float(musdl_cfg.get("score_min", 0.0))
        self.score_max = float(musdl_cfg.get("score_max", 1.0))

    def compute_loss(self, clips: torch.Tensor, targets: torch.Tensor,
                     epoch: int) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        logits = self.model(clips)  # [B, num_bins]
        loss = musdl_kl_loss(
            logits, targets,
            num_bins=self.num_bins,
            sigma=self.sigma,
            score_min=self.score_min,
            score_max=self.score_max,
        )
        # esperanza para logging diagnóstico (no entra al backward)
        with torch.no_grad():
            preds = expected_score(
                logits, num_bins=self.num_bins,
                score_min=self.score_min, score_max=self.score_max
            )
        return loss, {"loss_kl": loss.item()}, preds

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        preds, gts = [], []
        for batch in tqdm(self.val_loader, desc=f"epoch {epoch} val", leave=False):
            clips = batch["clip"].to(self.device, non_blocking=True)
            targets = batch["score"].to(self.device, non_blocking=True)
            with autocast(enabled=self.cfg.get("amp", True)):
                logits = self.model(clips)
                p = expected_score(
                    logits, num_bins=self.num_bins,
                    score_min=self.score_min, score_max=self.score_max,
                )
            preds.append(p.detach().float().cpu())
            gts.append(targets.detach().float().cpu())
        pred = torch.cat(preds).view(-1)
        gt = torch.cat(gts).view(-1)
        metrics = {
            "srcc": srcc(pred, gt),
            "plcc": plcc(pred, gt),
            "mae": mae(pred, gt, scale=self.cfg["data"].get("score_scale", 100.0)),
        }
        for k, v in metrics.items():
            self.tb.add_scalar(f"val/{k}", v, epoch)
        return metrics

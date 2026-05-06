"""Trainer para Coarse-to-Fine prototype head."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from src.models.prototype_head import assign_bins
from src.utils.metrics import mae, plcc, srcc

from .trainer import Trainer


class PrototypeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        proto_cfg = self.cfg.get("proto", {})
        self.num_coarse = int(proto_cfg.get("num_coarse", 10))
        self.num_fine = int(proto_cfg.get("num_fine", 5))
        self.score_min = float(proto_cfg.get("score_min", 0.0))
        self.score_max = float(proto_cfg.get("score_max", 1.0))
        self.alpha_fine = float(proto_cfg.get("alpha_fine", 0.5))

    def compute_loss(self, clips: torch.Tensor, targets: torch.Tensor,
                     epoch: int) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        out = self.model(clips)
        coarse_logits = out["coarse_logits"]                  # [B, K_c]
        fine_logits = out["fine_logits"]                      # [B, K_c, K_f]
        preds = out["score_pred"]                             # [B]

        c_idx, f_idx = assign_bins(targets, self.num_coarse, self.num_fine,
                                   self.score_min, self.score_max)
        l_coarse = F.cross_entropy(coarse_logits, c_idx)
        # fine: indexar logits[ batch, c_idx[batch], : ] vs f_idx[batch]
        b = coarse_logits.size(0)
        gather_idx = c_idx.view(b, 1, 1).expand(b, 1, self.num_fine)
        fine_logits_at_c = fine_logits.gather(1, gather_idx).squeeze(1)  # [B, K_f]
        l_fine = F.cross_entropy(fine_logits_at_c, f_idx)
        loss = l_coarse + self.alpha_fine * l_fine
        return loss, {
            "loss_total": loss.item(),
            "loss_coarse": l_coarse.item(),
            "loss_fine": l_fine.item(),
        }, preds

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        preds, gts = [], []
        for batch in tqdm(self.val_loader, desc=f"epoch {epoch} val", leave=False):
            clips = batch["clip"].to(self.device, non_blocking=True)
            targets = batch["score"].to(self.device, non_blocking=True)
            with autocast(enabled=self.cfg.get("amp", True)):
                out = self.model(clips)
                p = out["score_pred"]
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

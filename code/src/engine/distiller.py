"""Distiller: Trainer con pérdidas KD adicionales (regresión + atención +
alineación de features intermedias). El Teacher queda congelado."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from src.losses import FeatureAlignLoss, attention_kd_loss, regression_loss

from .trainer import Trainer


class Distiller(Trainer):
    def __init__(self,
                 student: nn.Module,
                 teacher: nn.Module,
                 cfg: Dict[str, Any],
                 train_loader,
                 val_loader,
                 run_dir):
        super().__init__(student, cfg, train_loader, val_loader, run_dir)
        self.teacher = teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Feature align: proyectar canales del Student y del Teacher a C común
        self.feat_align = FeatureAlignLoss(
            s_channels=student.mid_feat_channels,
            t_channels=teacher.mid_feat_channels,
            common=cfg["kd"].get("feat_align_common", 256),
        ).to(self.device)

        # añadir los parámetros del proyector al optimizador
        self.optimizer.add_param_group({"params": self.feat_align.parameters()})

        self.alpha = float(cfg["kd"]["alpha_reg"])
        self.beta = float(cfg["kd"]["beta_att"])
        self.gamma = float(cfg["kd"]["gamma_temp"])
        self.att_mode = cfg["kd"].get("att_loss", "kl")
        self.warmup_epochs = int(cfg["kd"].get("warmup_epochs", 5))
        self.freeze_bn = bool(cfg["kd"].get("freeze_bn", True))

    # ------------------------------------------------------------------
    def _set_student_bn_eval(self):
        """Congela estadísticos de BatchNorm del Student (running_mean/var)
        pero mantiene weights entrenables. Necesario cuando batch_size es
        muy pequeño (<4): con batch=1 o 2 las medias de lote son degenerativas
        y destruyen las features pre-entrenadas en ImageNet."""
        if not self.freeze_bn:
            return
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.SyncBatchNorm)):
                m.eval()

    def compute_loss(self, clips: torch.Tensor, targets: torch.Tensor,
                     epoch: int):
        # BN del Student en eval (ver docstring de _set_student_bn_eval).
        # Se llama en cada iteración porque Trainer.train_one_epoch() pone
        # model.train() antes de este compute_loss.
        self._set_student_bn_eval()

        # Teacher forward (congelado, AMP OK)
        with torch.no_grad():
            _ = self.teacher(clips)
        # Student forward
        s_pred = self.model(clips)

        # Pérdidas
        l_reg = regression_loss(s_pred, targets)
        feat_s = self.model.mid_feat
        feat_t = self.teacher.mid_feat.detach()

        l_att = attention_kd_loss(feat_s, feat_t, mode=self.att_mode)
        l_temp = self.feat_align(feat_s, feat_t)

        # warmup de β y γ: crecen linealmente durante las primeras epochs
        w = min(1.0, (epoch + 1) / max(self.warmup_epochs, 1))
        total = self.alpha * l_reg + w * (self.beta * l_att + self.gamma * l_temp)

        parts = {
            "loss": total.item(),
            "l_reg": l_reg.item(),
            "l_att": l_att.item(),
            "l_temp": l_temp.item(),
            "kd_weight": w,
        }
        return total, parts, s_pred

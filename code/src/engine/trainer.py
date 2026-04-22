"""Trainer base: fit loop con AMP, gradient accumulation, early stopping y
checkpoint del mejor modelo según SRCC de validación."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.metrics import mae, plcc, srcc


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 cfg: Dict[str, Any],
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 run_dir: Path):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda"))
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(enabled=cfg.get("amp", True))
        self.loss_fn = nn.MSELoss()

        self.tb = SummaryWriter(str(self.run_dir / "tb"))
        self.best_srcc = -1.0
        self.patience = 0
        self.global_step = 0

    # ------------------------------------------------------------------
    def _build_optimizer(self):
        tr = self.cfg["train"]
        lr = float(tr["lr"])
        wd = float(tr.get("weight_decay", 0.0))
        name = tr.get("optimizer", "adamw").lower()
        params = [p for p in self.model.parameters() if p.requires_grad]
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        raise ValueError(f"optimizer desconocido: {name}")

    def _build_scheduler(self):
        tr = self.cfg["train"]
        name = tr.get("scheduler", "cosine").lower()
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=tr["epochs"]
            )
        if name == "none":
            return None
        raise ValueError(f"scheduler desconocido: {name}")

    # ------------------------------------------------------------------
    def compute_loss(self, clips: torch.Tensor, targets: torch.Tensor,
                     epoch: int) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        """Pérdida por defecto: regresión MSE. Subclases pueden sobrescribir."""
        preds = self.model(clips)
        loss = self.loss_fn(preds.view(-1), targets.view(-1))
        return loss, {"loss": loss.item()}, preds

    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch: int):
        self.model.train()
        accum = int(self.cfg["train"].get("grad_accum_steps", 1))
        grad_clip = float(self.cfg["train"].get("grad_clip", 0.0))

        total = 0
        running = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(self.train_loader, desc=f"epoch {epoch} train", leave=False)
        for i, batch in enumerate(pbar):
            clips = batch["clip"].to(self.device, non_blocking=True)
            targets = batch["score"].to(self.device, non_blocking=True)

            with autocast(enabled=self.cfg.get("amp", True)):
                loss, parts, _ = self.compute_loss(clips, targets, epoch)
                loss_scaled = loss / accum

            self.scaler.scale(loss_scaled).backward()

            if (i + 1) % accum == 0:
                if grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running += loss.item() * clips.size(0)
            total += clips.size(0)

            self.global_step += 1
            if self.global_step % int(self.cfg["logging"].get("interval", 20)) == 0:
                for k, v in parts.items():
                    self.tb.add_scalar(f"train/{k}", v, self.global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if self.scheduler is not None:
            self.scheduler.step()
        avg = running / max(total, 1)
        self.tb.add_scalar("train/avg_loss", avg, epoch)
        return avg

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        preds, gts = [], []
        for batch in tqdm(self.val_loader, desc=f"epoch {epoch} val", leave=False):
            clips = batch["clip"].to(self.device, non_blocking=True)
            targets = batch["score"].to(self.device, non_blocking=True)
            with autocast(enabled=self.cfg.get("amp", True)):
                p = self.model(clips)
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

    # ------------------------------------------------------------------
    def save_checkpoint(self, name: str, metrics: Dict[str, float], epoch: int):
        ckpt = {
            "state_dict": self.model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "cfg": self.cfg,
        }
        torch.save(ckpt, self.run_dir / name)

    def fit(self):
        epochs = int(self.cfg["train"]["epochs"])
        patience_max = int(self.cfg["train"].get("early_stop_patience", 10))
        t0 = time.time()
        history = []

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(epoch)
            metrics = self.validate(epoch)
            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            print(f"[epoch {epoch:3d}] loss={train_loss:.4f}  "
                  f"val SRCC={metrics['srcc']:.4f}  PLCC={metrics['plcc']:.4f}  "
                  f"MAE={metrics['mae']:.3f}")

            if metrics["srcc"] > self.best_srcc:
                self.best_srcc = metrics["srcc"]
                self.patience = 0
                self.save_checkpoint("best.pth", metrics, epoch)
            else:
                self.patience += 1
                if self.patience >= patience_max:
                    print(f"Early stopping en epoch {epoch} (mejor SRCC={self.best_srcc:.4f})")
                    break

        elapsed = (time.time() - t0) / 60
        print(f"Entrenamiento terminado en {elapsed:.1f} min. Mejor SRCC val = {self.best_srcc:.4f}")
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        self.tb.close()

"""Evaluator: corre inferencia sobre un DataLoader y devuelve SRCC/PLCC/MAE."""
from __future__ import annotations

from typing import Dict

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import mae, plcc, srcc


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: str = "cuda",
             amp: bool = True,
             score_scale: float = 100.0) -> Dict[str, float]:
    model.to(device).eval()
    preds, gts = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        clips = batch["clip"].to(device, non_blocking=True)
        targets = batch["score"]
        with autocast(enabled=amp):
            p = model(clips)
        preds.append(p.detach().float().cpu())
        gts.append(targets.float())
    pred = torch.cat(preds).view(-1)
    gt = torch.cat(gts).view(-1)
    return {
        "srcc": srcc(pred, gt),
        "plcc": plcc(pred, gt),
        "mae":  mae(pred, gt, scale=score_scale),
        "n": int(pred.numel()),
    }

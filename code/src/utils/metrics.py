from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def srcc(pred, gt) -> float:
    p = _to_numpy(pred)
    g = _to_numpy(gt)
    if p.size < 2:
        return float("nan")
    return float(spearmanr(p, g).correlation)


def plcc(pred, gt) -> float:
    p = _to_numpy(pred)
    g = _to_numpy(gt)
    if p.size < 2:
        return float("nan")
    return float(pearsonr(p, g).statistic)


def mae(pred, gt, scale: float = 100.0) -> float:
    p = _to_numpy(pred)
    g = _to_numpy(gt)
    return float(np.abs(p - g).mean() * scale)

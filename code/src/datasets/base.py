"""Dataset base para AQA. Los subclases cargan splits y tensores pre-procesados."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .transforms import VideoTransform


@dataclass
class Sample:
    clip_id: str
    path: str
    score: float          # normalizado [0, 1]
    raw_score: float      # escala original
    category: str


class AQABaseDataset(Dataset):
    """
    Lee splits en JSON y devuelve (clip_tensor, score, meta).

    Split file format (JSON):
      [{ "clip_id": str, "path": str, "score": float, "raw_score": float, "category": str }, ...]

    Los tensores pre-procesados viven en `path` (relativo al root del repo code/).
    Shape en disco: [C, T, H, W] float16.
    """

    def __init__(self,
                 split_path: str | Path,
                 transform: Optional[VideoTransform] = None,
                 root: Optional[Path] = None):
        self.split_path = Path(split_path)
        self.root = root or self.split_path.parents[2]  # code/
        with open(self.split_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.samples: List[Sample] = [Sample(**s) for s in raw]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        clip_path = self.root / s.path
        clip = torch.load(clip_path, map_location="cpu").float()  # [C, T, H, W]
        if self.transform is not None:
            clip = self.transform(clip)
        return {
            "clip": clip,
            "score": torch.tensor(s.score, dtype=torch.float32),
            "meta": {
                "clip_id": s.clip_id,
                "category": s.category,
                "raw_score": s.raw_score,
            },
        }

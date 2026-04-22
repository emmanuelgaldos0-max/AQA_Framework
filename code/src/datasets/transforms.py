"""Transforms para clips de video en formato [C, T, H, W]."""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class VideoTransform:
    clip_length: int = 64
    frame_size: int = 224
    is_train: bool = False
    temporal_jitter: float = 0.1  # ±10%

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: [C, T, H, W] ya normalizado. Devuelve [C, clip_length, frame_size, frame_size]."""
        if self.is_train:
            if random.random() < 0.5:
                clip = torch.flip(clip, dims=[-1])

            T = clip.shape[1]
            jitter = int(T * self.temporal_jitter)
            if jitter > 0:
                start = random.randint(0, jitter)
                end = T - random.randint(0, jitter)
                if end - start >= 2:
                    clip = clip[:, start:end]

        # Ajustar a clip_length y frame_size por interpolación trilineal
        target_shape = (self.clip_length, self.frame_size, self.frame_size)
        if clip.shape[1:] != target_shape:
            clip = F.interpolate(
                clip.unsqueeze(0),
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        return clip

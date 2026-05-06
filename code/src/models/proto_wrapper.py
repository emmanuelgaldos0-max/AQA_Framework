"""Wrapper que reemplaza la cabeza de un Student por una PrototypeHead."""
from __future__ import annotations

import torch
import torch.nn as nn

from .prototype_head import PrototypeHead


class PrototypeWrapper(nn.Module):
    def __init__(self,
                 base_model: nn.Module,
                 num_coarse: int = 10,
                 num_fine: int = 5,
                 dropout: float = 0.5,
                 score_min: float = 0.0,
                 score_max: float = 1.0):
        super().__init__()
        self.base = base_model
        if not hasattr(base_model, "final_feat_channels"):
            raise AttributeError(
                "PrototypeWrapper requiere que el modelo base exponga `final_feat_channels`"
            )
        in_features = int(base_model.final_feat_channels)
        self.head = PrototypeHead(
            in_features=in_features,
            num_coarse=num_coarse,
            num_fine=num_fine,
            dropout=dropout,
            score_min=score_min,
            score_max=score_max,
        )
        self.base.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> dict:
        _ = self.base(x)
        feat = self.base.final_feat
        return self.head(feat)

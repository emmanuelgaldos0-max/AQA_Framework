from __future__ import annotations

from typing import Tuple

import torch


def count_flops(model: torch.nn.Module,
                input_shape: Tuple[int, ...] = (1, 3, 64, 224, 224),
                device: str = "cuda") -> float:
    """GFLOPs para un forward con el shape dado. Usa fvcore."""
    from fvcore.nn import FlopCountAnalysis

    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        total = FlopCountAnalysis(model, x).total()
    return float(total) / 1e9

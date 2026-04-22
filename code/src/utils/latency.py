from __future__ import annotations

import statistics
import time
from typing import Tuple

import torch


def benchmark_latency(model: torch.nn.Module,
                      input_shape: Tuple[int, ...] = (1, 3, 64, 224, 224),
                      warmup: int = 20,
                      runs: int = 100,
                      device: str = "cuda") -> float:
    """Latencia mediana en milisegundos por forward."""
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)

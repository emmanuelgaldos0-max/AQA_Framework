"""Mide FLOPs (vía fvcore) y latencia mediana (RTX 3060) para cada modelo.

Salida: JSON con {model_name: {params_M, flops_G, latency_ms_p50}}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import build_model


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def compute_flops(model, input_shape=(1, 3, 64, 224, 224)) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(*input_shape)
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
        model.eval()
        flops = FlopCountAnalysis(model, x).total() / 1e9
        return flops
    except Exception as e:
        print(f"  fvcore error: {e}")
        return -1.0


def measure_latency(model, input_shape=(1, 3, 64, 224, 224), n_warm=5, n_runs=20) -> float:
    if not torch.cuda.is_available():
        return -1.0
    model = model.cuda().eval()
    x = torch.randn(*input_shape).cuda()
    # warm
    with torch.no_grad():
        for _ in range(n_warm):
            _ = model(x)
    torch.cuda.synchronize()
    # timed
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]  # mediana


def main():
    models_to_eval = [
        ("i3d", {}),
        ("slowfast", {}),
        ("x3d_m", {}),
        ("tsm_mobilenetv2", {}),
        ("mobilenetv2_video", {}),
        ("mobilenetv3_large", {}),
    ]

    results = {}
    for name, kwargs in models_to_eval:
        print(f"\n=== {name} ===")
        try:
            m = build_model(name, clip_length=64, pretrained=False, **kwargs)
            params = count_params(m)
            print(f"  params: {params:.2f} M")
            flops = compute_flops(m)
            print(f"  FLOPs: {flops:.2f} G")
            lat = measure_latency(m)
            print(f"  latencia (p50): {lat:.1f} ms")
            results[name] = {"params_M": round(params, 2),
                             "flops_G": round(flops, 2),
                             "latency_ms_p50": round(lat, 1)}
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}

    out = Path(__file__).resolve().parents[1] / "experiments" / "efficiency_measurements.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado: {out}")
    print("\n=== RESUMEN ===")
    print(f"{'model':<22}{'Params (M)':>12}{'FLOPs (G)':>12}{'Latencia p50 (ms)':>20}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<22}  ERROR: {r['error']}")
        else:
            print(f"{name:<22}{r['params_M']:>12.2f}{r['flops_G']:>12.2f}{r['latency_ms_p50']:>20.1f}")


if __name__ == "__main__":
    main()

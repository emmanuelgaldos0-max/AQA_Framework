"""Genera un dataset sintético pequeño para probar el pipeline end-to-end
sin necesitar los datasets reales. 50 clips de video sintético en memoria
guardados directamente como tensores .pt.

Uso:
  python scripts/make_synthetic_dataset.py --n_clips 50 --dataset mit_diving_synth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clips", type=int, default=50)
    parser.add_argument("--dataset", default="mit_diving_synth")
    parser.add_argument("--clip_length", type=int, default=64)
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "preprocessed" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for i in range(args.n_clips):
        clip_id = f"synth_{i:04d}"
        # video sintético: patrones espacio-temporales correlacionados con el score
        base = torch.randn(3, args.clip_length, args.frame_size, args.frame_size) * 0.1
        raw_score = float(np.random.uniform(20, 95))
        # señal aprendible: amplitud proporcional al score
        t = torch.linspace(0, 3.14, args.clip_length).view(1, -1, 1, 1)
        modulation = torch.sin(t) * (raw_score / 100.0)
        clip = (base + modulation * 0.3).to(torch.float16)

        out_path = out_dir / f"{clip_id}.pt"
        torch.save(clip, out_path)

        index.append({
            "clip_id": clip_id,
            "path": str(out_path.relative_to(root)),
            "raw_score": raw_score,
            "score": raw_score / 100.0,
            "category": "synthetic",
        })

    idx_path = out_dir / "index.json"
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"Dataset sintético: {args.n_clips} clips en {out_dir}")
    print(f"Índice: {idx_path}")


if __name__ == "__main__":
    main()

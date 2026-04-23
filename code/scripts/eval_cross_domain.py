"""Evaluación cross-domain: carga checkpoints entrenados en un dataset
(source) y los evalúa sobre el test split de otro dataset (target) sin
fine-tuning, para medir generalización entre dominios.

Protocolo tabla 5.3 del PDF:
  - MTL-AQA → AQA-7 : diving especializado → multi-deporte
  - AQA-7 → JIGSAWS : deporte → cirugía robótica

Para cada transferencia y arquitectura (TSM-MBv2, MBv3):
  - Baseline (sin KD) en source → eval en target
  - KD en source → eval en target

Salida: JSON con todos los resultados + impresión en consola en formato tabla.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import VideoTransform, build_dataset
from src.engine import evaluate
from src.models import build_model
from src.utils.config import load_config
from src.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_ckpt(model: torch.nn.Module, path: Path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [ckpt] faltan {len(missing)} claves (primeras 3): {missing[:3]}")
    if unexpected:
        print(f"  [ckpt] sobran {len(unexpected)} claves (primeras 3): {unexpected[:3]}")


def run(source_ds: str, target_ds: str, arch_name: str, variant: str,
        ckpt_path: Path) -> dict:
    """arch_name ∈ {tsm_mobilenetv2, mobilenetv3_large}.
       variant ∈ {baseline, kd}."""
    if not ckpt_path.exists():
        print(f"  ! ckpt no existe: {ckpt_path}")
        return None

    # Config de referencia para dims / transforms
    cfg_name = {
        "tsm_mobilenetv2": "student_tsm_mbv2.yaml",
        "mobilenetv3_large": "student_mbv3.yaml",
    }[arch_name]
    cfg = load_config(ROOT / "configs" / cfg_name)

    set_seed(int(cfg.get("seed", 42)))
    clip_length = int(cfg["data"]["clip_length"])
    frame_size = int(cfg["data"]["frame_size"])
    eval_tf = VideoTransform(clip_length=clip_length, frame_size=frame_size, is_train=False)

    # Construir Student con el mismo clip_length
    model = build_model(arch_name, clip_length=clip_length, pretrained=False)
    _load_ckpt(model, ckpt_path)

    # Dataset target → test split
    test_ds = build_dataset(target_ds, "test", transform=eval_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
    )

    print(f"→ {arch_name} {variant}: {source_ds} → {target_ds}  "
          f"(test n={len(test_ds)})")
    metrics = evaluate(
        model, test_loader,
        device=DEVICE,
        amp=cfg.get("amp", True),
        score_scale=float(cfg["data"].get("score_scale", 100.0)),
    )
    print(f"   SRCC={metrics['srcc']:.4f}  PLCC={metrics['plcc']:.4f}  MAE={metrics['mae']:.3f}")
    return metrics


def main():
    results = []

    transfers = [
        ("mtl_aqa", "aqa7"),    # diving → multi-deporte
        ("aqa7",    "jigsaws"), # deporte → cirugía
    ]

    archs = ["tsm_mobilenetv2", "mobilenetv3_large"]

    for source, target in transfers:
        print("\n" + "=" * 60)
        print(f"Transferencia: {source} → {target}")
        print("=" * 60)
        for arch in archs:
            arch_short = {"tsm_mobilenetv2": "tsm_mbv2",
                          "mobilenetv3_large": "mbv3"}[arch]
            # Baseline
            baseline_ckpt = ROOT / "experiments" / f"{source}_student_{arch_short}_baseline_seed42" / "best.pth"
            m = run(source, target, arch, "baseline", baseline_ckpt)
            if m:
                results.append({
                    "source": source, "target": target,
                    "arch": arch, "variant": "baseline",
                    **{k: v for k, v in m.items() if k != "n"},
                    "n_test": m["n"],
                })

            # KD
            kd_ckpt = ROOT / "experiments" / f"{source}_{arch}_kd_seed42" / "best.pth"
            m = run(source, target, arch, "kd", kd_ckpt)
            if m:
                results.append({
                    "source": source, "target": target,
                    "arch": arch, "variant": "kd",
                    **{k: v for k, v in m.items() if k != "n"},
                    "n_test": m["n"],
                })

    # Imprimir resumen
    print("\n" + "=" * 72)
    print(f"{'source':<10s}{'target':<10s}{'arch':<22s}{'variant':<10s}{'SRCC':>8s}{'PLCC':>8s}{'MAE':>8s}")
    print("=" * 72)
    for r in results:
        print(f"{r['source']:<10s}{r['target']:<10s}{r['arch']:<22s}{r['variant']:<10s}"
              f"{r['srcc']:>8.4f}{r['plcc']:>8.4f}{r['mae']:>8.3f}")

    out = ROOT / "experiments" / "cross_domain_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()

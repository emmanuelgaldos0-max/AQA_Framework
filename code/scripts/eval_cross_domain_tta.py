"""Evaluación cross-domain extendida con Test-Time Adaptation (TTA).

Para cada par (source, target) y cada arquitectura {tsm_mobilenetv2,
mobilenetv3_large}, reporta:

    a) SRCC/PLCC/MAE sin TTA (baseline cross-domain).
    b) SRCC/PLCC/MAE con TTA-BN (BN-recalibration con clips del target train).

Se cubren los 6 pares posibles:
    aqa7 → mtl_aqa, aqa7 → jigsaws
    mtl_aqa → aqa7, mtl_aqa → jigsaws
    jigsaws → aqa7, jigsaws → mtl_aqa

Salida: JSON + tabla en consola.
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
from src.engine.tta_evaluator import evaluate_with_tta
from src.models import build_model
from src.utils.config import load_config
from src.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_ckpt(model: torch.nn.Module, path: Path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)


def run_pair(source: str, target: str, arch: str, ckpt: Path,
             tta_passes: int = 1, tta_reset: bool = True) -> dict:
    arch_short = {"tsm_mobilenetv2": "tsm_mbv2",
                  "mobilenetv3_large": "mbv3"}[arch]
    cfg_name = {"tsm_mobilenetv2": "student_tsm_mbv2.yaml",
                "mobilenetv3_large": "student_mbv3.yaml"}[arch]
    cfg = load_config(ROOT / "configs" / cfg_name)
    set_seed(int(cfg.get("seed", 42)))
    clip_length = int(cfg["data"]["clip_length"])
    frame_size = int(cfg["data"]["frame_size"])
    eval_tf = VideoTransform(clip_length=clip_length, frame_size=frame_size, is_train=False)
    score_scale = float(cfg["data"].get("score_scale", 100.0))
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg.get("num_workers", 4))

    # 1) split test del target (eval)
    test_ds = build_dataset(target, "test", transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    # 2) split train del target (adapt) – sólo para BN, sin labels
    adapt_ds = build_dataset(target, "train", transform=eval_tf)
    adapt_loader = DataLoader(adapt_ds, batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    print(f"\n→ {arch_short}: {source} → {target}  "
          f"(adapt n={len(adapt_ds)}, test n={len(test_ds)})")

    # --- (a) sin TTA ---
    model = build_model(arch, clip_length=clip_length, pretrained=False)
    _load_ckpt(model, ckpt)
    model = model.to(DEVICE)
    base = evaluate(model, test_loader, device=DEVICE, amp=cfg.get("amp", True),
                    score_scale=score_scale)
    print(f"   [no-TTA] SRCC={base['srcc']:.4f}  PLCC={base['plcc']:.4f}  MAE={base['mae']:.3f}")

    # --- (b) con TTA ---
    model_tta = build_model(arch, clip_length=clip_length, pretrained=False)
    _load_ckpt(model_tta, ckpt)
    model_tta = model_tta.to(DEVICE)
    tta = evaluate_with_tta(model_tta, adapt_loader, test_loader,
                            device=DEVICE, amp=cfg.get("amp", True),
                            score_scale=score_scale,
                            n_passes=tta_passes, reset_first=tta_reset)
    delta = tta["srcc"] - base["srcc"]
    print(f"   [BN-TTA] SRCC={tta['srcc']:.4f}  PLCC={tta['plcc']:.4f}  MAE={tta['mae']:.3f}  "
          f"Δ={delta:+.4f}")

    return {
        "source": source, "target": target, "arch": arch,
        "n_adapt": len(adapt_ds), "n_test": len(test_ds),
        "no_tta": {k: v for k, v in base.items() if k != "n"},
        "tta_bn": {k: v for k, v in tta.items() if k != "n"},
        "delta_srcc": delta,
    }


def main():
    results = []
    transfers = [
        ("aqa7",     "mtl_aqa"),
        ("aqa7",     "jigsaws"),
        ("mtl_aqa",  "aqa7"),
        ("mtl_aqa",  "jigsaws"),
        ("jigsaws",  "aqa7"),
        ("jigsaws",  "mtl_aqa"),
    ]
    archs = ["tsm_mobilenetv2", "mobilenetv3_large"]

    for source, target in transfers:
        print("\n" + "=" * 68)
        print(f"Transferencia: {source} → {target}")
        print("=" * 68)
        for arch in archs:
            arch_short = {"tsm_mobilenetv2": "tsm_mbv2",
                          "mobilenetv3_large": "mbv3"}[arch]
            ckpt = ROOT / "experiments" / f"{source}_student_{arch_short}_baseline_seed42" / "best.pth"
            if not ckpt.exists():
                print(f"  ! ckpt no existe: {ckpt}")
                continue
            results.append(run_pair(source, target, arch, ckpt))

    # tabla resumen
    print("\n" + "=" * 96)
    print(f"{'source':<10}{'target':<10}{'arch':<22}{'noTTA-SRCC':>12}{'TTA-SRCC':>12}"
          f"{'Δ':>10}")
    print("=" * 96)
    for r in results:
        print(f"{r['source']:<10}{r['target']:<10}{r['arch']:<22}"
              f"{r['no_tta']['srcc']:>12.4f}{r['tta_bn']['srcc']:>12.4f}"
              f"{r['delta_srcc']:>+10.4f}")

    out = ROOT / "experiments" / "cross_domain_tta_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()

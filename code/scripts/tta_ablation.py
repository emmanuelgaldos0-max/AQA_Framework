"""Ablación TTA-BN: comparar 1 pase vs 2 pases, con/sin reset de stats.
Sólo en 2 pares representativos (los de mejor y peor TTA en eval principal)
para no tomar demasiado tiempo.
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


def _load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)


def run_variant(source, target, arch, ckpt, n_passes, reset):
    cfg_name = {"tsm_mobilenetv2": "student_tsm_mbv2.yaml",
                "mobilenetv3_large": "student_mbv3.yaml"}[arch]
    cfg = load_config(ROOT / "configs" / cfg_name)
    set_seed(int(cfg.get("seed", 42)))
    clip_length = int(cfg["data"]["clip_length"])
    frame_size = int(cfg["data"]["frame_size"])
    eval_tf = VideoTransform(clip_length=clip_length, frame_size=frame_size, is_train=False)
    bs = int(cfg["train"]["batch_size"])

    test_ds = build_dataset(target, "test", transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    adapt_ds = build_dataset(target, "train", transform=eval_tf)
    adapt_loader = DataLoader(adapt_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(arch, clip_length=clip_length, pretrained=False).to(DEVICE)
    _load_ckpt(model, ckpt)
    m = evaluate_with_tta(model, adapt_loader, test_loader, device=DEVICE,
                          amp=cfg.get("amp", True), score_scale=float(cfg["data"].get("score_scale", 100.0)),
                          n_passes=n_passes, reset_first=reset)
    print(f"  passes={n_passes} reset={reset}: SRCC={m['srcc']:.4f}")
    return {"n_passes": n_passes, "reset": reset, "srcc": m["srcc"], "plcc": m["plcc"], "mae": m["mae"]}


def main():
    # 2 pares representativos: mejor (mtl→jigsaws MBv3 +0.54) y catastrófico (mtl→aqa7 MBv3 -0.68)
    cases = [
        ("mtl_aqa", "jigsaws", "mobilenetv3_large"),
        ("mtl_aqa", "aqa7", "mobilenetv3_large"),
    ]
    results = []
    for source, target, arch in cases:
        arch_short = {"tsm_mobilenetv2": "tsm_mbv2",
                      "mobilenetv3_large": "mbv3"}[arch]
        ckpt = ROOT / "experiments" / f"{source}_student_{arch_short}_baseline_seed42" / "best.pth"
        print(f"\n=== {source}→{target} {arch} ===")
        if not ckpt.exists():
            print(f"  ckpt no existe: {ckpt}"); continue
        for n_passes, reset in [(1, True), (2, True), (1, False), (2, False)]:
            r = run_variant(source, target, arch, ckpt, n_passes, reset)
            r.update({"source": source, "target": target, "arch": arch})
            results.append(r)

    out = ROOT / "experiments" / "tta_ablation.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()

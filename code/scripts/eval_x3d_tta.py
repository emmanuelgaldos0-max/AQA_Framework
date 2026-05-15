"""TTA cross-domain extendido a X3D-M.

Para los 6 pares (source, target) usa el checkpoint X3D-M entrenado en
source y evalúa en target con y sin TTA-BN (reset=True, 1 pase).
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


def run_pair(source, target, ckpt):
    cfg = load_config(ROOT / "configs" / "student_x3d_m.yaml")
    set_seed(42)
    clip_length = int(cfg["data"]["clip_length"])
    frame_size = int(cfg["data"]["frame_size"])
    eval_tf = VideoTransform(clip_length=clip_length, frame_size=frame_size, is_train=False)
    bs = int(cfg["train"]["batch_size"])

    test_ds = build_dataset(target, "test", transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    adapt_ds = build_dataset(target, "train", transform=eval_tf)
    adapt_loader = DataLoader(adapt_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n→ X3D-M: {source}→{target} (adapt n={len(adapt_ds)}, test n={len(test_ds)})")

    # sin TTA
    model = build_model("x3d_m", clip_length=clip_length, pretrained=False).to(DEVICE)
    _load_ckpt(model, ckpt)
    base = evaluate(model, test_loader, device=DEVICE, amp=cfg.get("amp", True),
                    score_scale=float(cfg["data"].get("score_scale", 100.0)))
    print(f"   [no-TTA] SRCC={base['srcc']:.4f}")

    # con TTA
    model_tta = build_model("x3d_m", clip_length=clip_length, pretrained=False).to(DEVICE)
    _load_ckpt(model_tta, ckpt)
    tta = evaluate_with_tta(model_tta, adapt_loader, test_loader,
                            device=DEVICE, amp=cfg.get("amp", True),
                            score_scale=float(cfg["data"].get("score_scale", 100.0)),
                            n_passes=1, reset_first=True)
    delta = tta["srcc"] - base["srcc"]
    print(f"   [BN-TTA] SRCC={tta['srcc']:.4f}  Δ={delta:+.4f}")

    return {"source": source, "target": target, "arch": "x3d_m",
            "no_tta": {k: v for k, v in base.items() if k != "n"},
            "tta_bn": {k: v for k, v in tta.items() if k != "n"},
            "delta_srcc": delta}


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
    for source, target in transfers:
        # checkpoints generados durante esta sesión
        ckpt = ROOT / "experiments" / f"{source}_x3d_m_seed42" / "best.pth"
        if not ckpt.exists():
            print(f"! ckpt no existe: {ckpt}"); continue
        results.append(run_pair(source, target, ckpt))

    out = ROOT / "experiments" / "x3d_tta_results.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGuardado: {out}")
    print()
    print(f"{'source':<10}{'target':<10}{'noTTA':>12}{'TTA':>12}{'Δ':>10}")
    for r in results:
        print(f"{r['source']:<10}{r['target']:<10}{r['no_tta']['srcc']:>12.4f}{r['tta_bn']['srcc']:>12.4f}{r['delta_srcc']:>+10.4f}")


if __name__ == "__main__":
    main()

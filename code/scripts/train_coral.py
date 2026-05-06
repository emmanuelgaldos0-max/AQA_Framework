"""Entry point para entrenar Students con domain adaptation CORAL.

Uso:
  # Source: AQA-7. Target: JIGSAWS (sin labels). Eval: split test JIGSAWS.
  python -m scripts.train_coral --config configs/coral_aqa7_to_jigsaws.yaml \
      --student mobilenetv3_large --source aqa7 --target jigsaws --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import VideoTransform, build_dataset
from src.engine.coral_trainer import CoralTrainer
from src.models import build_model
from src.utils.config import load_config
from src.utils.seed import set_seed


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--student", required=True,
                   choices=["mobilenetv3_large", "tsm_mobilenetv2"])
    p.add_argument("--source", required=True, choices=["aqa7", "mtl_aqa", "jigsaws"])
    p.add_argument("--target", required=True, choices=["aqa7", "mtl_aqa", "jigsaws"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def _build_loader(ds, cfg, shuffle: bool, batch_size: int = None) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size or int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=shuffle,
    )


def main():
    args = _parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    clip_length = int(cfg["data"]["clip_length"])
    frame_size = int(cfg["data"]["frame_size"])
    train_tf = VideoTransform(clip_length=clip_length, frame_size=frame_size, is_train=True)
    eval_tf = VideoTransform(clip_length=clip_length, frame_size=frame_size, is_train=False)

    model = build_model(args.student, clip_length=clip_length, pretrained=True)

    run_name = args.run_name or f"coral_{args.source}_to_{args.target}_{args.student}_seed{args.seed}"
    run_dir = Path(cfg["logging"]["checkpoint_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    src_train = build_dataset(args.source, "train", transform=train_tf)
    src_val = build_dataset(args.source, "val", transform=eval_tf)
    # target sin labels: usamos el split train del target (sus labels se ignoran).
    tgt_train = build_dataset(args.target, "train", transform=train_tf)
    # eval: split test del target
    tgt_test = build_dataset(args.target, "test", transform=eval_tf)
    print(f"[coral] src_train={len(src_train)}  src_val={len(src_val)}  "
          f"tgt_train={len(tgt_train)}  tgt_test={len(tgt_test)}")

    src_loader = _build_loader(src_train, cfg, shuffle=True)
    val_loader = _build_loader(src_val, cfg, shuffle=False)
    tgt_loader = _build_loader(tgt_train, cfg, shuffle=True)
    tgt_test_loader = _build_loader(tgt_test, cfg, shuffle=False)

    trainer = CoralTrainer(
        model, cfg, src_loader, val_loader, tgt_loader, run_dir
    )
    trainer.fit()

    # Evaluación final cross-domain sobre test split del target
    print("\n[coral] eval cross-domain sobre target test split:")
    from src.engine.evaluator import evaluate
    metrics = evaluate(
        model, tgt_test_loader,
        device=cfg.get("device", "cuda"),
        amp=cfg.get("amp", True),
        score_scale=float(cfg["data"].get("score_scale", 100.0)),
    )
    print(metrics)
    import json
    with open(run_dir / "cross_domain_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

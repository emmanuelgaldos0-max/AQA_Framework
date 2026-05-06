"""Entry point CLI para entrenar Students con MUSDL (Score Distribution Learning).

Uso:
  python -m scripts.train_musdl --config configs/musdl_aqa7_mbv3.yaml \
      --student mobilenetv3_large --dataset aqa7 --seed 42

Equivalente al baseline pero reemplazando la cabeza de regresión por una
cabeza de distribución y cambiando la pérdida a KL contra una gaussiana
discreta.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import VideoTransform, build_dataset
from src.engine.musdl_trainer import MUSDLTrainer
from src.models import build_model
from src.models.musdl_wrapper import MUSDLWrapper
from src.utils.config import load_config
from src.utils.seed import set_seed


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset", required=True,
                   choices=["aqa7", "mtl_aqa", "jigsaws"])
    p.add_argument("--student", required=True,
                   choices=["mobilenetv3_large", "tsm_mobilenetv2"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def _build_loader(ds, cfg, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
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

    base = build_model(args.student, clip_length=clip_length, pretrained=True)
    num_bins = int(cfg.get("musdl", {}).get("num_bins", 100))
    model = MUSDLWrapper(base, num_bins=num_bins,
                         dropout=float(cfg.get("musdl", {}).get("dropout", 0.5)))

    run_name = args.run_name or f"{args.dataset}_{args.student}_musdl_seed{args.seed}"
    run_dir = Path(cfg["logging"]["checkpoint_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset(args.dataset, "train", transform=train_tf)
    val_ds = build_dataset(args.dataset, "val", transform=eval_tf)
    print(f"[musdl] train: {len(train_ds)}  val: {len(val_ds)}  num_bins={num_bins}")
    train_loader = _build_loader(train_ds, cfg, shuffle=True)
    val_loader = _build_loader(val_ds, cfg, shuffle=False)

    trainer = MUSDLTrainer(model, cfg, train_loader, val_loader, run_dir)
    trainer.fit()


if __name__ == "__main__":
    main()

"""Entry point CLI para entrenar/evaluar modelos AQA.

Ejemplos:
  # Entrenar Teacher I3D sobre AQA-7
  python -m src.main --config configs/teacher_i3d.yaml --dataset aqa7 --seed 42

  # Baseline (Student sin KD)
  python -m src.main --config configs/student_tsm_mbv2.yaml --dataset aqa7 --seed 42

  # Student + KD (Teacher congelado desde checkpoint)
  python -m src.main --config configs/kd.yaml --student tsm_mobilenetv2 --dataset aqa7 \
      --teacher_ckpt experiments/teacher_aqa7_seed42/best.pth

  # Evaluación
  python -m src.main --mode eval --config configs/student_tsm_mbv2.yaml --dataset aqa7 \
      --ckpt experiments/student_aqa7_seed42/best.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import VideoTransform, build_dataset
from src.engine import Distiller, Trainer, evaluate
from src.models import build_model
from src.utils.config import load_config
from src.utils.seed import set_seed


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--dataset", required=True, choices=["aqa7", "mtl_aqa", "jigsaws"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--ckpt", type=Path, default=None, help="Checkpoint para evaluación o resume")
    p.add_argument("--teacher_ckpt", type=Path, default=None, help="Para KD")
    p.add_argument("--student", type=str, default=None,
                   help="Nombre del Student para el config kd.yaml (tsm_mobilenetv2 | mobilenetv3_large)")
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


def _load_checkpoint(model: torch.nn.Module, path: Path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ckpt] faltan {len(missing)} claves (primeras 3): {missing[:3]}")
    if unexpected:
        print(f"[ckpt] sobran {len(unexpected)} claves (primeras 3): {unexpected[:3]}")
    return ckpt


def main():
    args = _parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    clip_length = int(cfg["data"]["clip_length"])
    frame_size = int(cfg["data"]["frame_size"])
    train_tf = VideoTransform(clip_length=clip_length, frame_size=frame_size, is_train=True)
    eval_tf = VideoTransform(clip_length=clip_length, frame_size=frame_size, is_train=False)

    # ¿KD o sólo un modelo?
    is_kd = "kd" in cfg

    # Resolver modelo(s)
    if is_kd:
        student_name = args.student or cfg.get("model", {}).get("name")
        if not student_name:
            raise SystemExit("Para KD se requiere --student o `model.name` en el config.")
        student = build_model(student_name, clip_length=clip_length, pretrained=True)
        teacher = build_model("i3d", clip_length=clip_length, pretrained=False)
        teacher_ckpt = args.teacher_ckpt or cfg["kd"].get("teacher_ckpt")
        if teacher_ckpt:
            _load_checkpoint(teacher, Path(teacher_ckpt))
        model_name = f"kd_{student_name}"
    else:
        model_name = cfg["model"]["name"]
        model = build_model(model_name, clip_length=clip_length, pretrained=True)

    run_name = args.run_name or f"{args.dataset}_{model_name}_seed{args.seed}"
    run_dir = Path(cfg["logging"]["checkpoint_dir"]) / run_name

    # --------- modo eval ---------
    if args.mode == "eval":
        if not args.ckpt:
            raise SystemExit("--ckpt es obligatorio en modo eval")
        target_model = student if is_kd else model
        _load_checkpoint(target_model, args.ckpt)
        test_ds = build_dataset(args.dataset, "test", transform=eval_tf)
        test_loader = _build_loader(test_ds, cfg, shuffle=False)
        metrics = evaluate(
            target_model, test_loader,
            device=cfg.get("device", "cuda"),
            amp=cfg.get("amp", True),
            score_scale=float(cfg["data"].get("score_scale", 100.0)),
        )
        print(json.dumps(metrics, indent=2))
        return

    # --------- modo train ---------
    train_ds = build_dataset(args.dataset, "train", transform=train_tf)
    val_ds = build_dataset(args.dataset, "val", transform=eval_tf)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")
    train_loader = _build_loader(train_ds, cfg, shuffle=True)
    val_loader = _build_loader(val_ds, cfg, shuffle=False)

    if is_kd:
        engine = Distiller(student, teacher, cfg, train_loader, val_loader, run_dir)
    else:
        engine = Trainer(model, cfg, train_loader, val_loader, run_dir)

    engine.fit()


if __name__ == "__main__":
    main()

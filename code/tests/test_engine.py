"""Tests del Trainer y Distiller con dataset sintético chico."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@pytest.fixture(scope="module")
def synth_loaders():
    # Generar dataset sintético muy chico si no existe
    splits_dir = ROOT / "data" / "splits"
    if not (splits_dir / "synth_engine_train.json").exists():
        subprocess.run(
            [PY, "scripts/make_synthetic_dataset.py",
             "--n_clips", "40", "--dataset", "synth_engine", "--clip_length", "8", "--frame_size", "56"],
            cwd=ROOT, check=True, capture_output=True,
        )
        subprocess.run(
            [PY, "scripts/make_splits.py", "--dataset", "synth_engine", "--seed", "42"],
            cwd=ROOT, check=True, capture_output=True,
        )
    from src.datasets import AQABaseDataset, VideoTransform
    t = VideoTransform(clip_length=8, frame_size=56, is_train=False)
    train = AQABaseDataset(splits_dir / "synth_engine_train.json", transform=t, root=ROOT)
    val = AQABaseDataset(splits_dir / "synth_engine_val.json", transform=t, root=ROOT)
    return (
        DataLoader(train, batch_size=2, shuffle=True, num_workers=0),
        DataLoader(val, batch_size=2, shuffle=False, num_workers=0),
    )


def _minimal_cfg(tmp_path, kd: bool = False):
    cfg = {
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "amp": False,
        "num_workers": 0,
        "data": {"clip_length": 8, "frame_size": 56, "score_scale": 100.0},
        "train": {
            "batch_size": 2,
            "epochs": 2,
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "scheduler": "cosine",
            "grad_clip": 0.0,
            "early_stop_patience": 10,
            "grad_accum_steps": 1,
        },
        "logging": {"interval": 100, "checkpoint_dir": str(tmp_path)},
    }
    if kd:
        cfg["kd"] = {
            "alpha_reg": 1.0, "beta_att": 0.3, "gamma_temp": 0.3,
            "att_loss": "kl", "temp_loss": "mse",
            "warmup_epochs": 1, "feat_align_common": 64,
        }
    return cfg


def test_trainer_smoke(synth_loaders, tmp_path):
    from src.engine import Trainer
    from src.models import build_mobilenetv3

    cfg = _minimal_cfg(tmp_path)
    model = build_mobilenetv3(clip_length=8, pretrained=False, dropout=0.0)
    trainer = Trainer(model, cfg, *synth_loaders, run_dir=tmp_path / "run")
    trainer.fit()
    assert (tmp_path / "run" / "best.pth").exists()
    assert (tmp_path / "run" / "history.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="I3D es pesado; sólo en CUDA")
def test_distiller_smoke(synth_loaders, tmp_path):
    from src.engine import Distiller
    from src.models import build_i3d, build_tsm_mobilenetv2

    cfg = _minimal_cfg(tmp_path, kd=True)
    teacher = build_i3d(pretrained=False, dropout=0.0)
    student = build_tsm_mobilenetv2(clip_length=8, pretrained=False, dropout=0.0)
    d = Distiller(student, teacher, cfg, *synth_loaders, run_dir=tmp_path / "run_kd")
    d.fit()
    assert (tmp_path / "run_kd" / "best.pth").exists()

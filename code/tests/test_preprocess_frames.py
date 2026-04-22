"""Tests del pipeline de preprocesamiento con directorios de frames."""
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.datasets.preprocess import preprocess_video, read_frames_dir, read_video


def _make_fake_frames(dst: Path, n_frames: int = 60, size: int = 300):
    dst.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(dst / f"frame_{i:05d}.jpg"), img)


def test_read_frames_dir(tmp_path):
    d = tmp_path / "clip_001"
    _make_fake_frames(d, n_frames=30)
    frames = read_frames_dir(d, target_fps=25, src_fps=30)
    assert frames.ndim == 4
    assert frames.shape[-1] == 3
    assert frames.dtype == torch.uint8


def test_read_frames_dir_natural_order(tmp_path):
    d = tmp_path / "clip_002"
    d.mkdir()
    # números no-contiguos para forzar orden natural
    for i in [1, 2, 10, 20, 100]:
        img = np.full((8, 8, 3), i, dtype=np.uint8)
        cv2.imwrite(str(d / f"frame_{i}.jpg"), img)
    frames = read_frames_dir(d, target_fps=25, src_fps=25)  # sin submuestreo
    # el orden debe ser 1, 2, 10, 20, 100 (no lexicográfico)
    values = frames[:, 0, 0, 0].tolist()
    assert values == sorted(values)


def test_read_video_dispatches_directory(tmp_path):
    d = tmp_path / "clip_003"
    _make_fake_frames(d, n_frames=20)
    frames = read_video(d, target_fps=25, src_fps=30)
    assert frames.ndim == 4


def test_preprocess_video_from_frames(tmp_path):
    d = tmp_path / "clip_004"
    _make_fake_frames(d, n_frames=80, size=256)
    out = tmp_path / "out.pt"
    T, S = preprocess_video(d, out, clip_length=32, frame_size=224, target_fps=25, src_fps=30)
    assert T == 32 and S == 224
    assert out.exists()
    clip = torch.load(out, map_location="cpu")
    assert clip.shape == (3, 32, 224, 224)
    assert clip.dtype == torch.float16

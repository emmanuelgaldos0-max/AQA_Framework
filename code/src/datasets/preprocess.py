"""Funciones de preprocesamiento de video: extracción de frames, resize,
normalización y guardado como tensor .pt en float16."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import decord
    decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False

import cv2


KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


def read_video(path: str | Path, target_fps: int = 25) -> torch.Tensor:
    """Devuelve frames como tensor uint8 [T, H, W, 3] en RGB."""
    path = str(path)
    if _HAS_DECORD:
        vr = decord.VideoReader(path, num_threads=1)
        orig_fps = vr.get_avg_fps()
        step = max(1, int(round(orig_fps / target_fps)))
        idx = list(range(0, len(vr), step))
        frames = vr.get_batch(idx)  # torch tensor [T, H, W, 3]
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames.asnumpy())
        return frames
    # Fallback: OpenCV
    cap = cv2.VideoCapture(path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, int(round(orig_fps / target_fps)))
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"Video sin frames decodificables: {path}")
    return torch.from_numpy(np.stack(frames))


def resize_clip(frames: torch.Tensor, size: int = 224) -> torch.Tensor:
    """frames: [T, H, W, 3] uint8 → [T, size, size, 3] uint8 (resize bicúbico)."""
    T = frames.shape[0]
    out = np.empty((T, size, size, 3), dtype=np.uint8)
    arr = frames.numpy() if isinstance(frames, torch.Tensor) else frames
    for i in range(T):
        out[i] = cv2.resize(arr[i], (size, size), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(out)


def to_normalized_tensor(frames_uint8: torch.Tensor) -> torch.Tensor:
    """[T, H, W, 3] uint8 → [3, T, H, W] float32 normalizado con estadísticos Kinetics-400."""
    x = frames_uint8.to(torch.float32) / 255.0
    x = x.permute(3, 0, 1, 2).contiguous()  # [C, T, H, W]
    x = (x - KINETICS_MEAN) / KINETICS_STD
    return x


def pad_or_trim_temporal(clip: torch.Tensor, target_T: int) -> torch.Tensor:
    """Ajusta la dimensión temporal a target_T por padding (último frame) o recorte centrado."""
    T = clip.shape[1]
    if T == target_T:
        return clip
    if T < target_T:
        # pad por repetición del último frame
        pad = clip[:, -1:].expand(-1, target_T - T, -1, -1)
        return torch.cat([clip, pad], dim=1)
    # T > target_T → recorte centrado
    start = (T - target_T) // 2
    return clip[:, start:start + target_T]


def preprocess_video(
    input_path: str | Path,
    output_path: str | Path,
    clip_length: Optional[int] = None,
    frame_size: int = 224,
    target_fps: int = 25,
) -> Tuple[int, int]:
    """Lee video, normaliza y guarda como .pt float16. Devuelve (T_final, H=W=frame_size)."""
    frames = read_video(input_path, target_fps=target_fps)    # [T, H, W, 3] uint8
    frames = resize_clip(frames, size=frame_size)              # [T, s, s, 3]
    clip = to_normalized_tensor(frames)                        # [3, T, s, s] float32
    if clip_length is not None:
        clip = pad_or_trim_temporal(clip, clip_length)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clip.to(torch.float16), output_path)
    return int(clip.shape[1]), frame_size

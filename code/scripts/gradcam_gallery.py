"""Genera visualizaciones Grad-CAM para 5 clips representativos por dataset.

Protocolo §11.1 del plan:
  - 1 clip con score alto (top 10%)
  - 1 clip con score bajo (bottom 10%)
  - 3 clips con score medio
Comparativa por clip: Teacher I3D vs Student baseline vs Student+KD.

Salida: PNG en experiments/figures/gradcam/<dataset>/<clip_id>.png

Uso:
  python scripts/gradcam_gallery.py --datasets aqa7 mtl_aqa jigsaws
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import VideoTransform, build_dataset
from src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _denormalize(clip_chw: torch.Tensor) -> np.ndarray:
    """[C, T, H, W] normalizado → [T, H, W, 3] uint8 para visualización."""
    from src.datasets.preprocess import KINETICS_MEAN, KINETICS_STD
    clip = clip_chw.cpu().float() * KINETICS_STD + KINETICS_MEAN
    clip = clip.clamp(0, 1).permute(1, 2, 3, 0).numpy()
    return (clip * 255).astype(np.uint8)


def _attention_heatmap_from_feat(feat: torch.Tensor, clip_T: int = 64,
                                 target_shape=(224, 224)) -> np.ndarray:
    """[B=1, C, T', H, W] → [clip_T, 224, 224] heatmap normalizado [0,1].
    Interpola a la resolución espacial y temporal del clip original."""
    a = feat.float().pow(2).mean(dim=1)  # [1, T', H, W]
    a = F.interpolate(a.unsqueeze(1), size=(clip_T,) + target_shape,
                      mode="trilinear", align_corners=False).squeeze(1)  # [1, T, 224, 224]
    a = a.squeeze(0)
    a_min = a.amin(dim=(1, 2), keepdim=True)
    a_max = a.amax(dim=(1, 2), keepdim=True)
    a = (a - a_min) / (a_max - a_min + 1e-8)
    return a.cpu().numpy()


def _overlay(frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heat_rgb = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return ((1 - alpha) * frame_rgb + alpha * heat_rgb).astype(np.uint8)


def _load_model(arch: str, ckpt_path: Path, clip_length: int) -> torch.nn.Module:
    m = build_model(arch, clip_length=clip_length, pretrained=False)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu").get("state_dict", {})
        m.load_state_dict(state, strict=False)
    return m.to(DEVICE).eval()


def select_samples(ds, n_high=1, n_low=1, n_mid=3) -> List[int]:
    """Devuelve índices del dataset ordenados: [high, low, mid...]."""
    scores = np.array([s.score for s in ds.samples])
    idx_sorted = np.argsort(scores)
    idx_low = idx_sorted[:n_low]
    idx_high = idx_sorted[-n_high:]
    mid_start = len(idx_sorted) // 2 - n_mid // 2
    idx_mid = idx_sorted[mid_start:mid_start + n_mid]
    return list(idx_high) + list(idx_low) + list(idx_mid)


def generate_gallery(dataset: str, out_dir: Path):
    print(f"\n=== Grad-CAM gallery: {dataset} ===")
    transform = VideoTransform(clip_length=64, frame_size=224, is_train=False)
    ds = build_dataset(dataset, "test", transform=transform)

    # Modelos
    teacher = _load_model("i3d", ROOT / "experiments" / f"{dataset}_i3d_seed42" / "best.pth", clip_length=64)
    student_base = _load_model(
        "tsm_mobilenetv2",
        ROOT / "experiments" / f"{dataset}_student_tsm_mbv2_baseline_seed42" / "best.pth",
        clip_length=64,
    )
    student_kd = _load_model(
        "tsm_mobilenetv2",
        ROOT / "experiments" / f"{dataset}_tsm_mobilenetv2_kd_seed42" / "best.pth",
        clip_length=64,
    )

    sample_idx = select_samples(ds)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in sample_idx:
        sample = ds[idx]
        clip = sample["clip"].unsqueeze(0).to(DEVICE)
        clip_id = sample["meta"]["clip_id"]
        raw_score = sample["meta"]["raw_score"]

        # Inferir con cada modelo para obtener features
        with torch.no_grad():
            pred_t = teacher(clip).item() * 100
            feat_t = _attention_heatmap_from_feat(teacher.mid_feat)

            pred_b = student_base(clip).item() * 100
            feat_b = _attention_heatmap_from_feat(student_base.mid_feat)

            pred_kd = student_kd(clip).item() * 100
            feat_kd = _attention_heatmap_from_feat(student_kd.mid_feat)

        # Seleccionar el frame del medio como representativo
        frame_idx = 32
        frame_rgb = _denormalize(sample["clip"])[frame_idx]

        # Figura 1×4: original + Teacher + Student baseline + Student+KD
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(frame_rgb); axes[0].set_title(f"Frame original\nGT={raw_score:.1f}")
        axes[1].imshow(_overlay(frame_rgb, feat_t[frame_idx])); axes[1].set_title(f"Teacher I3D\npred={pred_t:.1f}")
        axes[2].imshow(_overlay(frame_rgb, feat_b[frame_idx])); axes[2].set_title(f"Student baseline\npred={pred_b:.1f}")
        axes[3].imshow(_overlay(frame_rgb, feat_kd[frame_idx])); axes[3].set_title(f"Student + KD\npred={pred_kd:.1f}")
        for ax in axes: ax.axis("off")
        plt.suptitle(f"{dataset} — {clip_id}", fontsize=11)
        plt.tight_layout()
        out_png = out_dir / f"{clip_id}.png"
        plt.savefig(out_png, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  {clip_id} (gt={raw_score:.1f})  T={pred_t:.1f} B={pred_b:.1f} KD={pred_kd:.1f}  → {out_png.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["aqa7", "mtl_aqa", "jigsaws"])
    args = parser.parse_args()

    out_root = ROOT / "experiments" / "figures" / "gradcam"
    for ds in args.datasets:
        generate_gallery(ds, out_root / ds)
    print(f"\nGrad-CAM gallery completa en {out_root}")


if __name__ == "__main__":
    main()

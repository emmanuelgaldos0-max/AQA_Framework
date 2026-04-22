"""Preprocesa todos los videos crudos de un dataset y genera un índice JSON.

Uso:
  python scripts/preprocess_videos.py --dataset mit_diving
  python scripts/preprocess_videos.py --dataset aqa7
  python scripts/preprocess_videos.py --dataset mtl_aqa
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# permitir ejecución como script sin instalar como paquete
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.preprocess import preprocess_video


def _load_annotations(dataset: str, raw_dir: Path) -> pd.DataFrame:
    """Devuelve DataFrame con columnas: clip_id, video_path, raw_score, category.

    `video_path` puede ser un archivo (.avi/.mp4) o un directorio de frames.
    """
    if dataset == "aqa7":
        scores_csv = raw_dir / "scores.csv"
        if not scores_csv.exists():
            raise FileNotFoundError(
                f"No existe {scores_csv}. Ejecutar scripts/build_aqa7_annotations.py primero."
            )
        df = pd.read_csv(scores_csv)  # clip_id, category, score, split, video_path
        df = df.rename(columns={"score": "raw_score"})
        # video_path ya viene como ruta absoluta al .avi (o a un dir de frames)
        return df

    if dataset in ("mtl_aqa", "jigsaws"):
        scores_csv = raw_dir / "scores.csv"
        if not scores_csv.exists():
            raise FileNotFoundError(
                f"No existe {scores_csv}. Ejecutar build_{dataset}_annotations.py primero."
            )
        df = pd.read_csv(scores_csv)
        # CSV ya trae: clip_id, category, raw_score (o score_raw), score, split, video_path
        if "score_raw" in df.columns and "raw_score" not in df.columns:
            df = df.rename(columns={"score_raw": "raw_score"})
        return df

    raise ValueError(f"Dataset desconocido: {dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["aqa7", "mtl_aqa", "jigsaws"])
    parser.add_argument("--clip_length", type=int, default=64)
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--src_fps", type=int, default=30,
                        help="FPS originales del directorio de frames (para submuestreo)")
    parser.add_argument("--score_scale", type=float, default=100.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw" / args.dataset
    out_dir = root / "data" / "preprocessed" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_annotations(args.dataset, raw_dir)
    print(f"[{args.dataset}] {len(df)} clips a procesar")

    index = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        clip_id = str(row["clip_id"])
        out_path = out_dir / f"{clip_id}.pt"
        if not out_path.exists():
            try:
                T, S = preprocess_video(
                    row["video_path"],
                    out_path,
                    clip_length=args.clip_length,
                    frame_size=args.frame_size,
                    target_fps=args.fps,
                    src_fps=args.src_fps,
                )
            except Exception as exc:
                print(f"  ! {clip_id}: {exc}")
                continue
        raw_score = float(row["raw_score"])
        index.append({
            "clip_id": clip_id,
            "path": str(out_path.relative_to(root)),
            "raw_score": raw_score,
            "score": raw_score / args.score_scale,
            "category": str(row["category"]),
        })

    idx_path = out_dir / "index.json"
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"Índice: {idx_path}  ({len(index)} clips OK)")


if __name__ == "__main__":
    main()

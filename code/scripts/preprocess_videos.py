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
    """Devuelve DataFrame con columnas: clip_id, video_path, raw_score, category."""
    if dataset == "mit_diving":
        scores_csv = raw_dir / "scores.csv"
        if not scores_csv.exists():
            raise FileNotFoundError(
                f"No existe {scores_csv}. Ejecutar scripts/download_datasets.sh y "
                "colocar anotaciones manualmente."
            )
        df = pd.read_csv(scores_csv)
        # Espera columnas video_id, score en [0, 100]
        videos_dir = raw_dir / "videos"
        df = df.rename(columns={"video_id": "clip_id", "score": "raw_score"})
        df["video_path"] = df["clip_id"].apply(lambda x: str(videos_dir / f"{x}.avi"))
        df["category"] = "diving"
        return df

    if dataset == "aqa7":
        # AQA-7 distribuye un .mat con splits; por ahora leemos un CSV plano.
        scores_csv = raw_dir / "scores.csv"
        if not scores_csv.exists():
            raise FileNotFoundError(
                f"No existe {scores_csv}. Generar desde el .mat oficial con un script auxiliar."
            )
        df = pd.read_csv(scores_csv)  # clip_id, category, score
        videos_dir = raw_dir / "videos"
        df = df.rename(columns={"score": "raw_score"})
        df["video_path"] = df.apply(
            lambda r: str(videos_dir / r["category"] / f"{r['clip_id']}.avi"), axis=1
        )
        return df

    if dataset == "mtl_aqa":
        ann = raw_dir / "annotations.json"
        if not ann.exists():
            raise FileNotFoundError(f"No existe {ann}")
        with open(ann, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        videos_dir = raw_dir / "videos"
        for clip_id, meta in data.items():
            rows.append({
                "clip_id": clip_id,
                "video_path": str(videos_dir / f"{clip_id}.mp4"),
                "raw_score": float(meta["final_score"]),
                "category": "diving",
            })
        return pd.DataFrame(rows)

    raise ValueError(f"Dataset desconocido: {dataset}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mit_diving", "aqa7", "mtl_aqa"])
    parser.add_argument("--clip_length", type=int, default=64)
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=25)
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

"""Parsea las anotaciones JIGSAWS distribuidas por MUSDL y genera:
  - data/raw/jigsaws/scores.csv   (clip_id, task, score, split, video_path)
  - data/splits/jigsaws_{train,val,test}.json

Los .pkl vienen de:
  https://raw.githubusercontent.com/nzl-thu/MUSDL/master/JIGSAWS/data/info/label.pkl
  https://raw.githubusercontent.com/nzl-thu/MUSDL/master/JIGSAWS/data/info/splits.pkl

label.pkl: dict { trial_id → [6 items OSATS cada uno 1..5] }
          GRS = sum(items) ∈ [6, 30]
          trial_id ejemplos: "Suturing_B001", "Knot_Tying_E003"
splits.pkl: dict { task → [4 folds], cada fold es list de trial_ids de TEST }
          (esquema 4-fold cross validation estándar en JIGSAWS)

Como los directorios de frames vienen como "<trial_id>_capture1" y
"<trial_id>_capture2" (dos vistas de la misma ejecución), generamos UN
clip_id por (trial, capture) y le asignamos la MISMA GRS.

Para integrarlo al pipeline 70/15/15 actual, consolidamos los 4 folds
de cada tarea y construimos un split estratificado por (tarea, bin de GRS).

Uso:
  python scripts/build_jigsaws_annotations.py \\
      --labels /tmp/jigsaws_label.pkl \\
      --splits /tmp/jigsaws_splits.pkl \\
      --frames_root Datasets/jigsaws_frames
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRS_MIN = 6
GRS_MAX = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Ruta al label.pkl de MUSDL")
    parser.add_argument("--splits", required=False, help="(no usado; reservado)")
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[1]
    frames_root = Path(args.frames_root).expanduser().resolve()

    with open(args.labels, "rb") as f:
        labels = pickle.load(f)  # dict: trial_id → [6 items]

    records = []
    missing = 0
    for trial_id, items in labels.items():
        task = trial_id.rsplit("_", 1)[0]   # "Needle_Passing_B001" → "Needle_Passing"
        grs = float(sum(items))              # 6..30
        grs_norm = (grs - GRS_MIN) / (GRS_MAX - GRS_MIN)  # → [0, 1]

        # Dos vistas por trial: capture1 y capture2
        for cap in ("capture1", "capture2"):
            clip_id = f"{trial_id}_{cap}"
            frames_dir = frames_root / clip_id
            if not frames_dir.exists():
                missing += 1
                continue
            records.append({
                "clip_id": clip_id,
                "category": task,
                "raw_score": grs,
                "score": grs_norm,
                "video_path": str(frames_dir),
                "trial_id": trial_id,
                "capture": cap,
            })

    print(f"JIGSAWS: {len(records)} clips (trial×vista) con frames (faltantes: {missing})")
    df = pd.DataFrame(records)

    # Estratificación por (tarea, bin de GRS)
    grs_bins = pd.cut(df["score"], bins=5, labels=False, duplicates="drop")
    strata = df["category"].astype(str) + "_" + grs_bins.astype(str)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=args.seed)
    idx_rest, idx_test = next(sss.split(df, strata))

    val_rel = args.val_ratio / (1.0 - args.test_ratio)
    df_rest = df.iloc[idx_rest].reset_index(drop=True)
    strata_rest = pd.Series(strata).iloc[idx_rest].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=args.seed)
    idx_tr, idx_val = next(sss2.split(df_rest, strata_rest))

    train_df = df_rest.iloc[idx_tr].copy(); train_df["split"] = "train"
    val_df = df_rest.iloc[idx_val].copy(); val_df["split"] = "val"
    test_df = df.iloc[idx_test].copy(); test_df["split"] = "test"
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    out_raw = code_root / "data" / "raw" / "jigsaws"
    out_raw.mkdir(parents=True, exist_ok=True)
    csv_path = out_raw / "scores.csv"
    all_df[["clip_id", "category", "raw_score", "score", "split", "video_path", "trial_id", "capture"]]\
        .to_csv(csv_path, index=False)

    out_splits = code_root / "data" / "splits"
    out_splits.mkdir(parents=True, exist_ok=True)
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rows = [{
            "clip_id": r.clip_id,
            "path": f"data/preprocessed/jigsaws/{r.clip_id}.pt",
            "raw_score": r.raw_score,
            "score": r.score,
            "category": r.category,
        } for r in sdf.itertuples()]
        path = out_splits / f"jigsaws_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"  {name}: {len(rows)} clips → {path}")

    print(f"CSV → {csv_path}")


if __name__ == "__main__":
    main()

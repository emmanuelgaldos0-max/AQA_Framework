"""Parsea el final_annotations_dict.pkl oficial de MTL-AQA y genera:
  - data/raw/mtl_aqa/scores.csv   (clip_id, score, split, video_path)
  - data/splits/mtl_aqa_{train,val,test}.json

El .pkl oficial vive en:
  https://raw.githubusercontent.com/ParitoshParmar/MTL-AQA/master/
  MTL-AQA_dataset_release/Ready_2_Use/MTL-AQA_split_0_data/final_annotations_dict.pkl

Las keys son tuplas (dive_id, clip_id). Los frames vienen en directorios
llamados "<dive_id:02d>_<clip_id:02d>".

Uso:
  python scripts/build_mtl_aqa_annotations.py \\
      --annotations /tmp/final_annotations_dict.pkl \\
      --frames_root Datasets/MTL_frames
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


def _key_to_folder(key: tuple[int, int]) -> str:
    return f"{key[0]:02d}_{key[1]:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True,
                        help="Ruta al final_annotations_dict.pkl oficial.")
    parser.add_argument("--frames_root", required=True,
                        help="Ruta raíz con los directorios <dive_id>_<clip_id>/ de frames.")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[1]
    with open(args.annotations, "rb") as f:
        ann = pickle.load(f)

    frames_root = Path(args.frames_root).expanduser().resolve()
    records = []
    missing = 0
    for key, meta in ann.items():
        folder = _key_to_folder(key)
        frames_dir = frames_root / folder
        if not frames_dir.exists():
            missing += 1
            continue
        records.append({
            "clip_id": folder,
            "category": "diving",
            "raw_score": float(meta["final_score"]),
            "score": float(meta["final_score"]) / 100.0,
            "video_path": str(frames_dir),
            "difficulty": float(meta.get("difficulty", 0.0)),
        })

    print(f"MTL-AQA: {len(records)} clips con frames (faltantes: {missing})")

    # Splits estratificados 70/15/15 por bins de score
    df = pd.DataFrame(records)
    bins = np.quantile(df["score"], np.linspace(0, 1, 6))
    strata = np.digitize(df["score"], bins[1:-1], right=True).astype(str)

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

    # CSV maestro
    out_raw = code_root / "data" / "raw" / "mtl_aqa"
    out_raw.mkdir(parents=True, exist_ok=True)
    csv_path = out_raw / "scores.csv"
    all_df[["clip_id", "category", "raw_score", "score", "split", "video_path"]]\
        .rename(columns={"raw_score": "score_raw"}).to_csv(csv_path, index=False)

    # JSON con paths a los .pt futuros
    out_splits = code_root / "data" / "splits"
    out_splits.mkdir(parents=True, exist_ok=True)
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rows = [{
            "clip_id": r.clip_id,
            "path": f"data/preprocessed/mtl_aqa/{r.clip_id}.pt",
            "raw_score": r.raw_score,
            "score": r.score,
            "category": r.category,
        } for r in sdf.itertuples()]
        path = out_splits / f"mtl_aqa_{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"  {name}: {len(rows)} clips → {path}")

    print(f"CSV → {csv_path}")


if __name__ == "__main__":
    main()

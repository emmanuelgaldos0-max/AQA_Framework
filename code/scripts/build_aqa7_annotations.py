"""Parsea los .mat oficiales de AQA-7 (Split-4) y genera:
  - data/raw/aqa7/scores.csv          (clip_id, category, score, split)
  - data/splits/aqa7_{train,val,test}.json  (split respetando el original)

El Split-4 oficial separa train (803) y test (~303). Se toma un 15% del
train como validación (estratificado por acción + bin de score, seed=42).

Uso:
  python scripts/build_aqa7_annotations.py --root Datasets/AQA-7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Mapping acción → carpeta (según Readme.txt de AQA-7)
ACTION_TO_CATEGORY = {
    1: "diving",
    2: "gym_vault",
    3: "ski_big_air",
    4: "snowboard_big_air",
    5: "sync_diving_3m",
    6: "sync_diving_10m",
}


def _load_mat(mat_path: Path) -> pd.DataFrame:
    m = loadmat(mat_path)
    # el nombre de la variable en el .mat
    key = [k for k in m if not k.startswith("__")][0]
    arr = np.asarray(m[key])
    df = pd.DataFrame(arr, columns=["action", "sample", "score"])
    df["action"] = df["action"].astype(int)
    df["sample"] = df["sample"].astype(int)
    return df


def _resolve_video_path(root: Path, action: int, sample: int) -> Path:
    category = ACTION_TO_CATEGORY[action]
    return root / "Actions" / category / f"{sample:03d}.avi"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Ruta al dataset AQA-7 (contiene Actions/, Split_4/).")
    parser.add_argument("--out_raw", default=None,
                        help="Directorio para scores.csv (por defecto code/data/raw/aqa7).")
    parser.add_argument("--out_splits", default=None,
                        help="Directorio para los JSON de splits (por defecto code/data/splits).")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Fracción del train oficial que se reserva para validación.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parents[1]
    root = Path(args.root).expanduser().resolve()
    out_raw = Path(args.out_raw) if args.out_raw else code_root / "data" / "raw" / "aqa7"
    out_splits = Path(args.out_splits) if args.out_splits else code_root / "data" / "splits"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_splits.mkdir(parents=True, exist_ok=True)

    train_df = _load_mat(root / "Split_4" / "split_4_train_list.mat")
    test_df = _load_mat(root / "Split_4" / "split_4_test_list.mat")

    def _to_records(df: pd.DataFrame, split: str) -> list[dict]:
        records = []
        for _, row in df.iterrows():
            category = ACTION_TO_CATEGORY[int(row["action"])]
            sample = int(row["sample"])
            video_path = _resolve_video_path(root, int(row["action"]), sample)
            if not video_path.exists():
                # el .mat puede contener augmentations duplicadas; saltar si falta
                continue
            raw_score = float(row["score"])
            records.append({
                "clip_id": f"{category}_{sample:03d}",
                "category": category,
                "video_path": str(video_path),
                "raw_score": raw_score,
                # escala AQA-7: los scores originales están en 0..~105; normalizamos por 100
                "score": raw_score / 100.0,
                "split": split,
            })
        return records

    train_records = _to_records(train_df, "train")
    test_records = _to_records(test_df, "test")

    # Deduplicar por clip_id (los .mat incluyen augmentations temporales duplicadas)
    def _dedup(rs: list[dict]) -> list[dict]:
        seen = {}
        for r in rs:
            seen.setdefault(r["clip_id"], r)
        return list(seen.values())

    train_records = _dedup(train_records)
    test_records = _dedup(test_records)
    print(f"AQA-7 Split-4: train únicos={len(train_records)}, test únicos={len(test_records)}")

    # Split train → train'/val estratificado por (categoria, bin de score)
    train_df2 = pd.DataFrame(train_records)
    bins = np.quantile(train_df2["score"], np.linspace(0, 1, 6))
    score_bin = np.digitize(train_df2["score"], bins[1:-1], right=True)
    strata = train_df2["category"].astype(str) + "_" + score_bin.astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    idx_train, idx_val = next(sss.split(train_df2, strata))
    new_train = [train_records[i] for i in idx_train]
    val_records = [train_records[i] for i in idx_val]
    print(f"  → train'={len(new_train)}  val={len(val_records)}  test={len(test_records)}")

    # CSV plano de scores + video_path (absoluto) para que preprocess lo use
    all_records = new_train + val_records + test_records
    csv_df = pd.DataFrame([{
        "clip_id": r["clip_id"],
        "category": r["category"],
        "score": r["raw_score"],
        "split": r["split"],
        "video_path": r["video_path"],
    } for r in all_records])
    csv_df.to_csv(out_raw / "scores.csv", index=False)

    # Splits JSON con paths (el preprocess los usará para saber qué .pt corresponden)
    # Pero aquí guardamos paths a los tensores .pt que generará preprocess_videos.py.
    def _to_split_json(records: list[dict], split: str):
        rows = []
        for r in records:
            pt_rel = f"data/preprocessed/aqa7/{r['clip_id']}.pt"
            rows.append({
                "clip_id": r["clip_id"],
                "path": pt_rel,
                "raw_score": r["raw_score"],
                "score": r["score"],
                "category": r["category"],
            })
        out = out_splits / f"aqa7_{split}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"  splits → {out}")

    _to_split_json(new_train, "train")
    _to_split_json(val_records, "val")
    _to_split_json(test_records, "test")

    print(f"CSV → {out_raw / 'scores.csv'}")


if __name__ == "__main__":
    main()

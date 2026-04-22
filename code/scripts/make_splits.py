"""Genera splits 70/15/15 estratificados a partir del índice preprocesado.

Estrategia: bineado del score en 5 cuantiles y StratifiedShuffleSplit.
Para AQA-7 se estratifica por (categoría, bin de score).

Uso:
  python scripts/make_splits.py --dataset mit_diving --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_bins(scores: np.ndarray, n_bins: int = 5) -> np.ndarray:
    edges = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    bins = np.digitize(scores, edges[1:-1], right=True)
    return bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    idx_path = root / "data" / "preprocessed" / args.dataset / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"Falta {idx_path}. Ejecutar preprocess_videos.py primero.")

    with open(idx_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    df = pd.DataFrame(index)

    scores = df["score"].to_numpy()
    score_bins = _make_bins(scores, n_bins=5)
    if "category" in df.columns and df["category"].nunique() > 1:
        strata = df["category"].astype(str) + "_" + score_bins.astype(str)
    else:
        strata = pd.Series(score_bins.astype(str))

    # Paso 1: test vs rest
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=args.seed)
    idx_rest, idx_test = next(sss.split(df, strata))

    # Paso 2: val vs train dentro de rest
    val_relative = args.val_ratio / (1.0 - args.test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=args.seed)
    df_rest = df.iloc[idx_rest].reset_index(drop=True)
    strata_rest = strata.iloc[idx_rest].reset_index(drop=True)
    idx_train_rel, idx_val_rel = next(sss2.split(df_rest, strata_rest))

    splits = {
        "train": df_rest.iloc[idx_train_rel].to_dict(orient="records"),
        "val":   df_rest.iloc[idx_val_rel].to_dict(orient="records"),
        "test":  df.iloc[idx_test].to_dict(orient="records"),
    }

    out_dir = root / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        out_path = out_dir / f"{args.dataset}_{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"{split_name:5} → {out_path.name}  ({len(rows)} clips)")


if __name__ == "__main__":
    main()

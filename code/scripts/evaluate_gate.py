"""Evalúa el gate de decisión de la fase 8 a partir de los logs de
entrenamiento. Compara los mejores SRCC de los runs MUSDL contra los
baselines guardados en experiments/all_results.json.

Reglas del gate (de PLAN_FASE_8.md):
  - Pasa si ≥ 2/4 mejoran ≥ +0.020 SRCC sobre baseline.
  - Pasa si ≥ 1/4 mejora ≥ +0.05 SRCC sobre baseline.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASELINES = {
    ("aqa7", "mobilenetv3_large"): 0.8854,
    ("aqa7", "tsm_mobilenetv2"): 0.8968,
    ("mtl_aqa", "mobilenetv3_large"): 0.8703,
    ("mtl_aqa", "tsm_mobilenetv2"): 0.8804,
}

LOG_FILES = {
    ("aqa7", "mobilenetv3_large"): "experiments/musdl_logs/aqa7_mbv3.log",
    ("aqa7", "tsm_mobilenetv2"): "experiments/musdl_logs/aqa7_tsm.log",
    ("mtl_aqa", "mobilenetv3_large"): "experiments/musdl_logs/mtl_mbv3.log",
    ("mtl_aqa", "tsm_mobilenetv2"): "experiments/musdl_logs/mtl_tsm.log",
}


def best_srcc_from_log(path: Path) -> float | None:
    """Extrae el mejor SRCC del log (línea 'Mejor SRCC val' o máximo de val SRCC=)."""
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    # 1) línea final del trainer
    m = re.search(r"Mejor SRCC val\s*=\s*([0-9.]+)", txt)
    if m:
        return float(m.group(1))
    # 2) máximo de val SRCC=
    matches = re.findall(r"val SRCC=([0-9.]+)", txt)
    if matches:
        return max(float(x) for x in matches)
    return None


def main():
    rows = []
    for (dataset, student), baseline in BASELINES.items():
        log_path = ROOT / LOG_FILES[(dataset, student)]
        srcc = best_srcc_from_log(log_path)
        delta = (srcc - baseline) if srcc is not None else None
        rows.append({
            "dataset": dataset,
            "student": student,
            "baseline": baseline,
            "musdl_best": srcc,
            "delta": delta,
            "log": str(log_path.relative_to(ROOT)),
        })

    print(f"{'dataset':<10} {'student':<22} {'baseline':>9} {'musdl':>9} {'Δ':>9}")
    for r in rows:
        srcc = f"{r['musdl_best']:.4f}" if r['musdl_best'] is not None else "—"
        delta = f"{r['delta']:+.4f}" if r['delta'] is not None else "—"
        print(f"{r['dataset']:<10} {r['student']:<22} {r['baseline']:>9.4f} {srcc:>9} {delta:>9}")

    # Reglas gate
    have = [r for r in rows if r["delta"] is not None]
    n_pass_020 = sum(1 for r in have if r["delta"] >= 0.020)
    any_pass_050 = any(r["delta"] >= 0.050 for r in have)
    print()
    print(f"Configs ≥ +0.020 SRCC: {n_pass_020}/4")
    print(f"Algún config ≥ +0.050 SRCC: {any_pass_050}")

    gate_pass = (n_pass_020 >= 2) or any_pass_050
    print()
    print("GATE CAMINO 1:", "PASA ✅" if gate_pass else "FALLA ❌")

    out = ROOT / "experiments" / "musdl_logs" / "gate_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"rows": rows, "gate_pass": gate_pass,
                   "n_pass_020": n_pass_020, "any_pass_050": any_pass_050}, f, indent=2)
    print(f"\nResumen guardado en {out.relative_to(ROOT)}")
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()

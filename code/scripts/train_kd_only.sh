#!/usr/bin/env bash
# Entrena los 6 Students+KD (2 arq × 3 datasets × 1 semilla=42).
# Los baselines y Teachers ya están entrenados.

set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
SEED=42

run() {
    local desc="$1"; shift
    echo ""
    echo "============================================================"
    echo "[$(date)] $desc"
    echo "============================================================"
    "$@" || { echo "[$(date)] FALLÓ: $desc" >&2; return 1; }
    echo "[$(date)] OK: $desc"
}

for ds in aqa7 mtl_aqa jigsaws; do
    TEACHER_CKPT="experiments/${ds}_i3d_seed${SEED}/best.pth"
    for student in tsm_mobilenetv2 mobilenetv3_large; do
        run "Student+KD $student $ds seed=$SEED" \
            $PY -m src.main \
                --config configs/kd.yaml \
                --dataset $ds \
                --seed $SEED \
                --student $student \
                --teacher_ckpt "$TEACHER_CKPT" \
                --run_name "${ds}_${student}_kd_seed${SEED}"
    done
done

echo ""
echo "============================================================"
echo "[$(date)] KD pipeline COMPLETO (6 runs, 1 semilla)."
echo "============================================================"

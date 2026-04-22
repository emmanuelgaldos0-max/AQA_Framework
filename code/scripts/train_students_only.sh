#!/usr/bin/env bash
# Entrena los 6 baselines + 18 KD runs (24 total) con los Teachers ya
# entrenados en experiments/<dataset>_i3d_seed42/best.pth.
#
# Configs ajustados a batch=2 + grad_accum=8 para Students y batch=1 +
# grad_accum=16 para KD, requeridos por los 6 GB de la RTX 3060 con T=64.

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

# ---------- Students baseline ----------
for ds in aqa7 mtl_aqa jigsaws; do
    for cfg in student_tsm_mbv2 student_mbv3; do
        run "Student baseline $cfg $ds" \
            $PY -m src.main \
                --config configs/${cfg}.yaml \
                --dataset $ds \
                --seed $SEED \
                --run_name "${ds}_${cfg}_baseline_seed${SEED}"
    done
done

# ---------- Students + KD (3 semillas) ----------
for ds in aqa7 mtl_aqa jigsaws; do
    TEACHER_CKPT="experiments/${ds}_i3d_seed${SEED}/best.pth"
    for student in tsm_mobilenetv2 mobilenetv3_large; do
        for seed in 42 1337 2024; do
            run "Student+KD $student $ds seed=$seed" \
                $PY -m src.main \
                    --config configs/kd.yaml \
                    --dataset $ds \
                    --seed $seed \
                    --student $student \
                    --teacher_ckpt "$TEACHER_CKPT" \
                    --run_name "${ds}_${student}_kd_seed${seed}"
        done
    done
done

echo ""
echo "============================================================"
echo "[$(date)] Students + KD pipeline COMPLETO."
echo "============================================================"

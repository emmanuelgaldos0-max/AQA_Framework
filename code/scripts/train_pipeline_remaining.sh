#!/usr/bin/env bash
# Pipeline encadenado de entrenamientos.
#
# Espera a que termine el Teacher AQA-7 que está corriendo, luego ejecuta
# en secuencia:
#   1. Teacher I3D en MTL-AQA y JIGSAWS
#   2. Students baseline (TSM-MBv2, MBv3) en los 3 datasets
#   3. Students + KD con 3 semillas (42, 1337, 2024) en los 3 datasets
#
# Total: 2 Teachers + 6 baselines + 18 KD runs = 26 entrenamientos.
# Tiempo estimado: 15-20 horas.
#
# Uso:
#   nohup bash scripts/train_pipeline_remaining.sh > /tmp/pipeline.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
SEED=42

TEACHER_LOG="/tmp/claude-1000/-home-sam-Documentos-Github-Personal-Tesis/b2bfb28e-bfbd-43ae-98a4-5c4b32114f44/tasks/bn7li43hl.output"

echo "[$(date)] Esperando a que termine Teacher AQA-7..."
until [ -f "$TEACHER_LOG" ] && grep -qE "Entrenamiento terminado|Early stopping" "$TEACHER_LOG" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] Teacher AQA-7 listo. Iniciando pipeline."

run() {
    local desc="$1"; shift
    echo ""
    echo "============================================================"
    echo "[$(date)] $desc"
    echo "============================================================"
    "$@" || { echo "[$(date)] FALLÓ: $desc" >&2; return 1; }
    echo "[$(date)] OK: $desc"
}

# ---------- 1. Teachers restantes ----------
for ds in mtl_aqa jigsaws; do
    run "Teacher I3D $ds" \
        $PY -m src.main \
            --config configs/teacher_i3d.yaml \
            --dataset $ds \
            --seed $SEED \
            --run_name "${ds}_i3d_seed${SEED}"
done

# ---------- 2. Students baseline (sin KD) ----------
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

# ---------- 3. Students + KD (3 semillas por combinación) ----------
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
echo "[$(date)] Pipeline COMPLETO."
echo "============================================================"

#!/bin/bash
# Cadena Fase 9 — robustez empírica alineada con plan del asesor.
#   E5 × 4 (semillas 0,7 × 2 students) — AQA-7
#   E6 × 1 (sin ImageNet pretrain) — TSM-MBv2 AQA-7
#   E7 × 1 (sin TSM) — MBv2 plano AQA-7
#   E1+ × 1 (Teacher SlowFast-R50) — AQA-7

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOGS=experiments/phase9_logs
mkdir -p "$LOGS"
ts() { date '+%H:%M:%S'; }

# ----------- E5: réplicas semillas 0 y 7, 2 students en AQA-7 ----------
echo "[$(ts)] E5 — TSM-MBv2 baseline AQA-7 seed 0"
python -m src.main --config configs/student_tsm_mbv2.yaml --dataset aqa7 \
    --seed 0 --run_name aqa7_tsm_mbv2_baseline_seed0 > "$LOGS/E5_tsm_seed0.log" 2>&1

echo "[$(ts)] E5 — TSM-MBv2 baseline AQA-7 seed 7"
python -m src.main --config configs/student_tsm_mbv2.yaml --dataset aqa7 \
    --seed 7 --run_name aqa7_tsm_mbv2_baseline_seed7 > "$LOGS/E5_tsm_seed7.log" 2>&1

echo "[$(ts)] E5 — MBv3 baseline AQA-7 seed 0"
python -m src.main --config configs/student_mbv3.yaml --dataset aqa7 \
    --seed 0 --run_name aqa7_mbv3_baseline_seed0 > "$LOGS/E5_mbv3_seed0.log" 2>&1

echo "[$(ts)] E5 — MBv3 baseline AQA-7 seed 7"
python -m src.main --config configs/student_mbv3.yaml --dataset aqa7 \
    --seed 7 --run_name aqa7_mbv3_baseline_seed7 > "$LOGS/E5_mbv3_seed7.log" 2>&1

# ----------- E6: TSM-MBv2 sin pretrain, AQA-7 -------------------------
echo "[$(ts)] E6 — TSM-MBv2 AQA-7 NO pretrain (seed 42)"
python -m src.main --config configs/student_tsm_mbv2_nopretrain.yaml --dataset aqa7 \
    --seed 42 --run_name aqa7_tsm_mbv2_nopretrain_seed42 > "$LOGS/E6_tsm_nopretrain.log" 2>&1

# ----------- E7: MBv2 plano (sin TSM), AQA-7 --------------------------
echo "[$(ts)] E7 — MBv2 plano (sin TSM) AQA-7 (seed 42)"
python -m src.main --config configs/student_mbv2_video.yaml --dataset aqa7 \
    --seed 42 --run_name aqa7_mbv2_video_seed42 > "$LOGS/E7_mbv2_notsm.log" 2>&1

# ----------- E1 extendido: Teacher SlowFast-R50 AQA-7 -----------------
echo "[$(ts)] E1+ — Teacher SlowFast-R50 AQA-7 (seed 42)"
python -m src.main --config configs/teacher_slowfast.yaml --dataset aqa7 \
    --seed 42 --run_name aqa7_slowfast_seed42 > "$LOGS/E1plus_slowfast.log" 2>&1

echo "[$(ts)] PHASE 9 CHAIN DONE"

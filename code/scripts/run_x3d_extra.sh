#!/bin/bash
# Encadena X3D-M en MTL-AQA y JIGSAWS para completar tabla SOTA.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
LOGS=experiments/phase10_logs
mkdir -p "$LOGS"
ts() { date '+%H:%M:%S'; }

echo "[$(ts)] X3D-M MTL-AQA"
python -m src.main --config configs/x3d_m_mtl.yaml --dataset mtl_aqa --seed 42 \
    --run_name mtl_aqa_x3d_m_seed42 > "$LOGS/x3d_m_mtl_aqa.log" 2>&1

echo "[$(ts)] X3D-M JIGSAWS"
python -m src.main --config configs/x3d_m_jigsaws.yaml --dataset jigsaws --seed 42 \
    --run_name jigsaws_x3d_m_seed42 > "$LOGS/x3d_m_jigsaws.log" 2>&1

echo "[$(ts)] X3D EXTRA DONE"

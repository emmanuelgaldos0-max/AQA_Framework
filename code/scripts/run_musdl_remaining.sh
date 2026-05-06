#!/bin/bash
# Encadena los 3 entrenamientos MUSDL restantes (después de MBv3+AQA-7).
# Sólo se lanza si MBv3+AQA-7 cumplió el gate de Camino 1.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOGS=experiments/musdl_logs
mkdir -p "$LOGS"

echo "[$(date '+%H:%M:%S')] starting MUSDL TSM-MBv2 + AQA-7"
python -m scripts.train_musdl --config configs/musdl_aqa7_tsm.yaml \
    --student tsm_mobilenetv2 --dataset aqa7 --seed 42 \
    --run_name aqa7_tsm_mbv2_musdl_seed42 > "$LOGS/aqa7_tsm.log" 2>&1

echo "[$(date '+%H:%M:%S')] starting MUSDL MBv3 + MTL-AQA"
python -m scripts.train_musdl --config configs/musdl_mtl_mbv3.yaml \
    --student mobilenetv3_large --dataset mtl_aqa --seed 42 \
    --run_name mtl_aqa_mbv3_musdl_seed42 > "$LOGS/mtl_mbv3.log" 2>&1

echo "[$(date '+%H:%M:%S')] starting MUSDL TSM-MBv2 + MTL-AQA"
python -m scripts.train_musdl --config configs/musdl_mtl_tsm.yaml \
    --student tsm_mobilenetv2 --dataset mtl_aqa --seed 42 \
    --run_name mtl_aqa_tsm_mbv2_musdl_seed42 > "$LOGS/mtl_tsm.log" 2>&1

echo "[$(date '+%H:%M:%S')] all 3 MUSDL runs done"

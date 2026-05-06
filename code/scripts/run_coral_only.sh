#!/bin/bash
# Re-lanza las 4 corridas CORAL con batch=1, grad_accum=16, BN frozen.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOGS=experiments/coral_logs
mkdir -p "$LOGS"
ts() { date '+%H:%M:%S'; }

echo "[$(ts)] CORAL aqa7 -> jigsaws (MBv3)"
python -m scripts.train_coral --config configs/coral_aqa7_to_jigsaws.yaml \
    --student mobilenetv3_large --source aqa7 --target jigsaws --seed 42 \
    --run_name coral_aqa7_to_jigsaws_mbv3_seed42 > "$LOGS/aqa7_to_jigsaws_mbv3.log" 2>&1

echo "[$(ts)] CORAL aqa7 -> jigsaws (TSM-MBv2)"
python -m scripts.train_coral --config configs/coral_aqa7_to_jigsaws_tsm.yaml \
    --student tsm_mobilenetv2 --source aqa7 --target jigsaws --seed 42 \
    --run_name coral_aqa7_to_jigsaws_tsm_seed42 > "$LOGS/aqa7_to_jigsaws_tsm.log" 2>&1

echo "[$(ts)] CORAL mtl_aqa -> aqa7 (MBv3)"
python -m scripts.train_coral --config configs/coral_mtl_to_aqa7.yaml \
    --student mobilenetv3_large --source mtl_aqa --target aqa7 --seed 42 \
    --run_name coral_mtl_to_aqa7_mbv3_seed42 > "$LOGS/mtl_to_aqa7_mbv3.log" 2>&1

echo "[$(ts)] CORAL mtl_aqa -> aqa7 (TSM-MBv2)"
python -m scripts.train_coral --config configs/coral_mtl_to_aqa7_tsm.yaml \
    --student tsm_mobilenetv2 --source mtl_aqa --target aqa7 --seed 42 \
    --run_name coral_mtl_to_aqa7_tsm_seed42 > "$LOGS/mtl_to_aqa7_tsm.log" 2>&1

echo "[$(ts)] CORAL CHAIN DONE"

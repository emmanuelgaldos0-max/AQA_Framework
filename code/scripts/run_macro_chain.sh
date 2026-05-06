#!/bin/bash
# Macro-cadena fase 8.2:
#   * 2 corridas MUSDL JIGSAWS (MBv3, TSM)
#   * 4 corridas CoFInAl (intra-domain: AQA-7, MTL-AQA × MBv3, TSM)
#   * 4 corridas CORAL (cross-domain: aqa7→jigsaws, mtl→aqa7 × MBv3, TSM)
#
# Cada bloque escribe a su propio log. Al final imprime "MACRO CHAIN DONE".

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

MUSDL_LOGS=experiments/musdl_logs
PROTO_LOGS=experiments/proto_logs
CORAL_LOGS=experiments/coral_logs
mkdir -p "$MUSDL_LOGS" "$PROTO_LOGS" "$CORAL_LOGS"

ts() { date '+%H:%M:%S'; }

# ---------------------------- MUSDL JIGSAWS (2) ----------------------------
echo "[$(ts)] MUSDL JIGSAWS MBv3"
python -m scripts.train_musdl --config configs/musdl_jigsaws_mbv3.yaml \
    --student mobilenetv3_large --dataset jigsaws --seed 42 \
    --run_name jigsaws_mbv3_musdl_seed42 > "$MUSDL_LOGS/jigsaws_mbv3.log" 2>&1

echo "[$(ts)] MUSDL JIGSAWS TSM-MBv2"
python -m scripts.train_musdl --config configs/musdl_jigsaws_tsm.yaml \
    --student tsm_mobilenetv2 --dataset jigsaws --seed 42 \
    --run_name jigsaws_tsm_mbv2_musdl_seed42 > "$MUSDL_LOGS/jigsaws_tsm.log" 2>&1

# ---------------------------- CoFInAl (4) ---------------------------------
echo "[$(ts)] CoFInAl MBv3 + AQA-7"
python -m scripts.train_proto --config configs/proto_aqa7_mbv3.yaml \
    --student mobilenetv3_large --dataset aqa7 --seed 42 \
    --run_name aqa7_mbv3_proto_seed42 > "$PROTO_LOGS/aqa7_mbv3.log" 2>&1

echo "[$(ts)] CoFInAl TSM-MBv2 + AQA-7"
python -m scripts.train_proto --config configs/proto_aqa7_tsm.yaml \
    --student tsm_mobilenetv2 --dataset aqa7 --seed 42 \
    --run_name aqa7_tsm_mbv2_proto_seed42 > "$PROTO_LOGS/aqa7_tsm.log" 2>&1

echo "[$(ts)] CoFInAl MBv3 + MTL-AQA"
python -m scripts.train_proto --config configs/proto_mtl_mbv3.yaml \
    --student mobilenetv3_large --dataset mtl_aqa --seed 42 \
    --run_name mtl_aqa_mbv3_proto_seed42 > "$PROTO_LOGS/mtl_mbv3.log" 2>&1

echo "[$(ts)] CoFInAl TSM-MBv2 + MTL-AQA"
python -m scripts.train_proto --config configs/proto_mtl_tsm.yaml \
    --student tsm_mobilenetv2 --dataset mtl_aqa --seed 42 \
    --run_name mtl_aqa_tsm_mbv2_proto_seed42 > "$PROTO_LOGS/mtl_tsm.log" 2>&1

# ---------------------------- CORAL (4) -----------------------------------
echo "[$(ts)] CORAL aqa7 -> jigsaws (MBv3)"
python -m scripts.train_coral --config configs/coral_aqa7_to_jigsaws.yaml \
    --student mobilenetv3_large --source aqa7 --target jigsaws --seed 42 \
    --run_name coral_aqa7_to_jigsaws_mbv3_seed42 > "$CORAL_LOGS/aqa7_to_jigsaws_mbv3.log" 2>&1

echo "[$(ts)] CORAL aqa7 -> jigsaws (TSM-MBv2)"
python -m scripts.train_coral --config configs/coral_aqa7_to_jigsaws_tsm.yaml \
    --student tsm_mobilenetv2 --source aqa7 --target jigsaws --seed 42 \
    --run_name coral_aqa7_to_jigsaws_tsm_seed42 > "$CORAL_LOGS/aqa7_to_jigsaws_tsm.log" 2>&1

echo "[$(ts)] CORAL mtl_aqa -> aqa7 (MBv3)"
python -m scripts.train_coral --config configs/coral_mtl_to_aqa7.yaml \
    --student mobilenetv3_large --source mtl_aqa --target aqa7 --seed 42 \
    --run_name coral_mtl_to_aqa7_mbv3_seed42 > "$CORAL_LOGS/mtl_to_aqa7_mbv3.log" 2>&1

echo "[$(ts)] CORAL mtl_aqa -> aqa7 (TSM-MBv2)"
python -m scripts.train_coral --config configs/coral_mtl_to_aqa7_tsm.yaml \
    --student tsm_mobilenetv2 --source mtl_aqa --target aqa7 --seed 42 \
    --run_name coral_mtl_to_aqa7_tsm_seed42 > "$CORAL_LOGS/mtl_to_aqa7_tsm.log" 2>&1

echo "[$(ts)] MACRO CHAIN DONE"

#!/bin/bash
# Lanza CORAL para 2 transferencias × 2 students × 3 lambdas = 12 corridas.
# Sólo si el camino 1 (MUSDL) NO cumplió gate.

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

LOGS=experiments/coral_logs
mkdir -p "$LOGS"

run_coral() {
    local cfg=$1
    local source=$2
    local target=$3
    local student=$4
    local lambda=$5
    local tag="${source}_to_${target}_${student}_lam${lambda}"
    echo "[$(date '+%H:%M:%S')] CORAL $tag"
    # Override λ vía env var (config se lee con default)
    CORAL_LAMBDA="$lambda" python -m scripts.train_coral \
        --config "configs/$cfg" --source "$source" --target "$target" \
        --student "$student" --seed 42 \
        --run_name "coral_$tag" > "$LOGS/$tag.log" 2>&1
}

# AQA-7 → JIGSAWS (peor caso actual: SRCC ≈ 0)
for student in mobilenetv3_large tsm_mobilenetv2; do
    for lambda in 0.01 0.1 1.0; do
        run_coral coral_aqa7_to_jigsaws.yaml aqa7 jigsaws "$student" "$lambda"
    done
done

# MTL-AQA → AQA-7 (parcial: ≈ 0.55)
for student in mobilenetv3_large tsm_mobilenetv2; do
    for lambda in 0.01 0.1 1.0; do
        run_coral coral_mtl_to_aqa7.yaml mtl_aqa aqa7 "$student" "$lambda"
    done
done

echo "[$(date '+%H:%M:%S')] CORAL: 12 runs done"

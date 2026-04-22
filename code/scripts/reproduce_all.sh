#!/usr/bin/env bash
# Reproduce el pipeline completo: anotaciones → preprocess → entrenamientos → evaluación.
#
# Requiere:
#   - .venv activado
#   - Datasets en ../Datasets/{AQA-7, MTL_frames, jigsaws_frames}
#   - Anotaciones auxiliares en /tmp: final_annotations_dict.pkl, jigsaws_label.pkl
#     (se descargan automáticamente en build_aqa7_annotations.py / MUSDL)

set -e
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
SEED=42
DATASETS_ROOT="../../Datasets"   # relativo a code/

echo "=========================================="
echo " 1. Anotaciones"
echo "=========================================="
$PY scripts/build_aqa7_annotations.py --root "$DATASETS_ROOT/AQA-7"
$PY scripts/build_mtl_aqa_annotations.py \
    --annotations /tmp/final_annotations_dict.pkl \
    --frames_root "$DATASETS_ROOT/MTL_frames"
$PY scripts/build_jigsaws_annotations.py \
    --labels /tmp/jigsaws_label.pkl \
    --splits /tmp/jigsaws_splits.pkl \
    --frames_root "$DATASETS_ROOT/jigsaws_frames"

echo "=========================================="
echo " 2. Preprocesamiento (decode + resize + normalize)"
echo "=========================================="
for ds in aqa7 mtl_aqa jigsaws; do
  $PY scripts/preprocess_videos.py --dataset $ds --clip_length 64 --frame_size 224 --fps 25 --src_fps 30
done

echo "=========================================="
echo " 3. Entrenar Teacher I3D por dataset"
echo "=========================================="
for ds in aqa7 mtl_aqa jigsaws; do
  $PY -m src.main --config configs/teacher_i3d.yaml --dataset $ds --seed $SEED
done

echo "=========================================="
echo " 4. Entrenar Students baseline (sin KD)"
echo "=========================================="
for ds in aqa7 mtl_aqa jigsaws; do
  for cfg in student_tsm_mbv2 student_mbv3; do
    $PY -m src.main --config configs/${cfg}.yaml --dataset $ds --seed $SEED
  done
done

echo "=========================================="
echo " 5. Entrenar Students + KD (3 semillas)"
echo "=========================================="
for ds in aqa7 mtl_aqa jigsaws; do
  for student in tsm_mobilenetv2 mobilenetv3_large; do
    for seed in 42 1337 2024; do
      TEACHER_CKPT="experiments/${ds}_i3d_seed${SEED}/best.pth"
      $PY -m src.main \
          --config configs/kd.yaml \
          --dataset $ds \
          --seed $seed \
          --student $student \
          --teacher_ckpt "$TEACHER_CKPT"
    done
  done
done

echo "=========================================="
echo " 6. Evaluación final"
echo "=========================================="
# Placeholder: el evaluator se corre implícitamente después de fit().
# Los ckpts best.pth de cada run contienen las métricas de validación.
# Un script de agregación de tabla consolidada está pendiente.

echo "Pipeline reproducible completado."

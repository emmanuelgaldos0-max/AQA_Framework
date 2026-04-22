#!/usr/bin/env bash
#
# Descarga los datasets de AQA utilizados en la tesis.
#
# Los tres datasets vienen como directorios de frames pre-extraídos
# (no archivos de video).
#
# Requisitos: gdown (pip install gdown) y acceso interactivo a los repos
# oficiales para aceptar términos.
#
# Uso:
#   bash scripts/download_datasets.sh aqa7
#   bash scripts/download_datasets.sh mtl_aqa
#   bash scripts/download_datasets.sh jigsaws
#   bash scripts/download_datasets.sh all

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"

mkdir -p "$RAW"/{aqa7,mtl_aqa,jigsaws}

TARGET="${1:-all}"

download_aqa7() {
  local dir="$RAW/aqa7"
  echo "[AQA-7] → $dir"
  if [ -f "$dir/.downloaded" ]; then
    echo "  ya descargado"; return 0
  fi
  cat <<EOF
  Instrucciones:
    1. Ir a https://github.com/ParitoshParmar/Action-Quality-Assessment-AQA-7-
    2. Seguir el link de Google Drive con los frames pre-extraídos
    3. Descomprimir en:
         $dir/frames/<category>/<clip_id>/frame_00001.jpg ...
    4. Colocar el CSV de scores en:
         $dir/scores.csv   (columnas: clip_id, category, score)
    5. Crear el marcador:
         touch $dir/.downloaded
EOF
}

download_mtl_aqa() {
  local dir="$RAW/mtl_aqa"
  echo "[MTL-AQA] → $dir"
  if [ -f "$dir/.downloaded" ]; then
    echo "  ya descargado"; return 0
  fi
  cat <<EOF
  Instrucciones:
    1. Ir a https://github.com/ParitoshParmar/MTL-AQA
    2. Descargar el pack de frames pre-extraídos desde Google Drive
    3. Descomprimir en:
         $dir/frames/<clip_id>/frame_00001.jpg ...
    4. Colocar las anotaciones en:
         $dir/annotations.json   (por clip_id → {final_score, subscores...})
    5. Crear el marcador:
         touch $dir/.downloaded
EOF
}

download_jigsaws() {
  local dir="$RAW/jigsaws"
  echo "[JIGSAWS] → $dir"
  if [ -f "$dir/.downloaded" ]; then
    echo "  ya descargado"; return 0
  fi
  cat <<EOF
  Instrucciones:
    1. Solicitar acceso en: https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/
    2. Aceptar los términos y descargar los .zip de Suturing, Needle_Passing y Knot_Tying
    3. Descomprimir en:
         $dir/frames/<task>/<clip_id>/frame_00001.jpg ...
       (las distribuciones oficiales traen frames en carpetas por trial)
    4. Convertir las etiquetas OSATS a CSV:
         $dir/scores.csv   (columnas: clip_id, task, score)
       (puntuación global = suma de los 6 items OSATS, rango 6..30)
    5. Crear el marcador:
         touch $dir/.downloaded
EOF
}

case "$TARGET" in
  aqa7)    download_aqa7 ;;
  mtl_aqa) download_mtl_aqa ;;
  jigsaws) download_jigsaws ;;
  all)
    download_aqa7 || true
    download_mtl_aqa || true
    download_jigsaws || true
    ;;
  *)
    echo "Uso: $0 {aqa7|mtl_aqa|jigsaws|all}"
    exit 1
    ;;
esac

echo "Listo."

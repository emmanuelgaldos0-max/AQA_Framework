#!/usr/bin/env bash
#
# Descarga los datasets de AQA (MIT-Diving, AQA-7, MTL-AQA).
#
# Los enlaces "oficiales" de estos datasets son histó­ricamente inestables:
# MIT-Diving vive en la página del autor (Pirsiavash, MIT CSAIL); AQA-7 y
# MTL-AQA viven en Google Drive gestionados por Paritosh Parmar.
#
# Requisitos: wget, gdown (pip install gdown).
#
# Uso:
#   bash scripts/download_datasets.sh mit_diving
#   bash scripts/download_datasets.sh aqa7
#   bash scripts/download_datasets.sh mtl_aqa
#   bash scripts/download_datasets.sh all
#
# Las URLs de Drive pueden caducar. Si alguna falla, verifica en:
#   https://github.com/ParitoshParmar/MTL-AQA
#   https://github.com/ParitoshParmar/Action-Quality-Assessment-AQA-7-

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"

mkdir -p "$RAW"/{mit_diving,aqa7,mtl_aqa}

TARGET="${1:-all}"

download_mit_diving() {
  local dir="$RAW/mit_diving"
  echo "[MIT-Diving] → $dir"

  # URL histórica del paper ECCV 2014 (Pirsiavash et al.).
  # En la práctica suele venir como parte de MTL-AQA (diving subset).
  # Estrategia: avisar y dejar al usuario que coloque los videos.
  if [ ! -f "$dir/.downloaded" ]; then
    cat <<EOF
  ⚠  MIT-Diving no tiene enlace directo estable. Opciones:
     1. Descargar desde el subset "diving" de MTL-AQA (bash scripts/download_datasets.sh mtl_aqa).
     2. Contactar a Pirsiavash vía emmanuelgaldos0@gmail.com para el dataset original.
     3. Buscar mirror en HuggingFace: huggingface.co/datasets (query "mit diving").

  Cuando tengas los videos, colócalos en:
     $dir/videos/
  y las puntuaciones en:
     $dir/scores.csv   (columnas: video_id,score)
EOF
  else
    echo "  ya descargado (marcador .downloaded presente)"
  fi
}

download_aqa7() {
  local dir="$RAW/aqa7"
  echo "[AQA-7] → $dir"

  if [ -f "$dir/.downloaded" ]; then
    echo "  ya descargado"
    return 0
  fi

  # ID de Google Drive del archivo oficial según README de Parmar.
  # Si este ID cambia, actualizar en https://github.com/ParitoshParmar/Action-Quality-Assessment-AQA-7-
  local drive_id="1yfNVHk2a-o1dsA_bIhk2s0sRJRzs6tHy"
  pushd "$dir" > /dev/null
  "$ROOT/.venv/bin/gdown" --id "$drive_id" -O AQA-7.zip || {
    echo "  ❌ gdown falló. Verifica el ID en el repo oficial o usa el navegador."
    popd > /dev/null
    return 1
  }
  unzip -q AQA-7.zip && rm AQA-7.zip
  touch .downloaded
  popd > /dev/null
}

download_mtl_aqa() {
  local dir="$RAW/mtl_aqa"
  echo "[MTL-AQA] → $dir"

  if [ -f "$dir/.downloaded" ]; then
    echo "  ya descargado"
    return 0
  fi

  # Repo: github.com/ParitoshParmar/MTL-AQA
  # El README indica un Drive folder. ID a confirmar cuando se ejecute.
  echo "  ⚠  MTL-AQA requiere aceptar los términos en el README del repo."
  echo "     Ir a: https://github.com/ParitoshParmar/MTL-AQA"
  echo "     Seguir las instrucciones de descarga."
  echo "     Colocar los videos en: $dir/videos/"
  echo "     Anotaciones en: $dir/annotations.json"
}

case "$TARGET" in
  mit_diving) download_mit_diving ;;
  aqa7)       download_aqa7 ;;
  mtl_aqa)    download_mtl_aqa ;;
  all)
    download_mit_diving || true
    download_aqa7 || true
    download_mtl_aqa || true
    ;;
  *)
    echo "Uso: $0 {mit_diving|aqa7|mtl_aqa|all}"
    exit 1
    ;;
esac

echo "Listo."

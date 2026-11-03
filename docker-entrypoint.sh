#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${FOCUSFLEET_DATA_DIR:-/data}"
PORT="${PORT:-7860}"

mkdir -p "${DATA_DIR}/logs"

seed_asset() {
  local name="$1"
  if [[ -e "${DATA_DIR}/${name}" ]]; then
    return 0
  fi
  for src in "/app/runtime-assets/${name}" "/app/${name}"; do
    if [[ -e "${src}" ]]; then
      cp -a "${src}" "${DATA_DIR}/${name}"
      return 0
    fi
  done
}

seed_asset "driver_drowsiness_model.keras"
seed_asset "mi-gente-sountec-live-edit.mp3"
seed_asset "driver.db"

HAAR_DST="${DATA_DIR}/haarcascade_frontalface_default.xml"
if [[ ! -f "${HAAR_DST}" ]]; then
  HAAR_SRC="$(python -c "import cv2, os; print(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))")"
  if [[ -f "${HAAR_SRC}" ]]; then
    cp "${HAAR_SRC}" "${HAAR_DST}"
  else
    curl -fsSL -o "${HAAR_DST}" \
      "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
  fi
fi
export FOCUSFLEET_HAAR_CASCADE="${HAAR_DST}"

export FOCUSFLEET_DATA_DIR="${DATA_DIR}"
export PYTHONPATH="/app/backend:/app:${PYTHONPATH:-}"

cd "${DATA_DIR}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --app-dir /app/backend

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

export FOCUSFLEET_DATA_DIR="${DATA_DIR}"
export PYTHONPATH="/app/backend:/app:${PYTHONPATH:-}"

cd "${DATA_DIR}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --app-dir /app/backend

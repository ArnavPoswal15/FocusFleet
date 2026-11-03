#!/usr/bin/env bash
# Sync GitHub monorepo → Hugging Face Space (Docker SDK layout).
#
# Usage:
#   ./scripts/sync-huggingface-space.sh              # rsync + commit + push HF main
#   ./scripts/sync-huggingface-space.sh --dry-run    # show rsync changes only
#
# Override HF clone path:
#   HF_SPACE_DIR=/path/to/FocusFleet ./scripts/sync-huggingface-space.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB_DIR="$REPO_ROOT/web"
HF_DIR="${HF_SPACE_DIR:-$HOME/Desktop/huggingface/FocusFleet}"
DRY_RUN=false
PUSH=true

RSYNC_EXCLUDES=(
  --exclude='.git/'
  --exclude='__pycache__/'
  --exclude='.DS_Store'
  --exclude='venv/'
  --exclude='*.code-workspace'
  --exclude='.hf-sync-meta.json'
  --exclude='node_modules/'
  --exclude='.next/'
  --exclude='out/'
)

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true; PUSH=false ;;
    --no-push) PUSH=false ;;
    -h|--help)
      sed -n '2,11p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

for path in "$WEB_DIR" "$REPO_ROOT/Dockerfile" "$REPO_ROOT/docker-entrypoint.sh"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

if [[ ! -d "$HF_DIR/.git" ]]; then
  echo "HF Space clone not found (expected git repo): $HF_DIR" >&2
  echo "Clone with: git clone https://huggingface.co/spaces/ArnavPoswal15/FocusFleet \"$HF_DIR\"" >&2
  exit 1
fi

RSYNC_FLAGS=(-av)
if $DRY_RUN; then
  RSYNC_FLAGS+=(-n --itemize-changes)
fi

mkdir -p "$HF_DIR/runtime-assets"

# Do not use --delete: HF may hold LFS assets (e.g. *.keras) not tracked in GitHub.
rsync "${RSYNC_FLAGS[@]}" "${RSYNC_EXCLUDES[@]}" \
  "$REPO_ROOT/focusfleet/" "$HF_DIR/focusfleet/"

rsync "${RSYNC_FLAGS[@]}" "${RSYNC_EXCLUDES[@]}" \
  "$REPO_ROOT/backend/" "$HF_DIR/backend/"

rsync "${RSYNC_FLAGS[@]}" "${RSYNC_EXCLUDES[@]}" \
  "$REPO_ROOT/frontend/" "$HF_DIR/frontend/"

rsync "${RSYNC_FLAGS[@]}" \
  "$REPO_ROOT/Dockerfile" \
  "$REPO_ROOT/docker-entrypoint.sh" \
  "$HF_DIR/"

if [[ -f "$REPO_ROOT/runtime-assets/mi-gente-sountec-live-edit.mp3" ]]; then
  rsync "${RSYNC_FLAGS[@]}" \
    "$REPO_ROOT/runtime-assets/" \
    "$HF_DIR/runtime-assets/"
elif [[ -f "$WEB_DIR/mi-gente-sountec-live-edit.mp3" ]]; then
  rsync "${RSYNC_FLAGS[@]}" \
    "$WEB_DIR/mi-gente-sountec-live-edit.mp3" \
    "$HF_DIR/runtime-assets/"
fi

if [[ -f "$WEB_DIR/driver_drowsiness_model.keras" ]]; then
  rsync "${RSYNC_FLAGS[@]}" \
    "$WEB_DIR/driver_drowsiness_model.keras" \
    "$HF_DIR/runtime-assets/"
fi

if [[ -f "$WEB_DIR/driver.db" ]]; then
  rsync "${RSYNC_FLAGS[@]}" \
    "$WEB_DIR/driver.db" \
    "$HF_DIR/runtime-assets/"
fi

if [[ -f "$WEB_DIR/.gitattributes" ]]; then
  rsync "${RSYNC_FLAGS[@]}" \
    "$WEB_DIR/.gitattributes" \
    "$HF_DIR/"
fi

if [[ -f "$WEB_DIR/README-DOCKER.md" ]]; then
  if $DRY_RUN; then
    echo "Would copy README-DOCKER.md → README.md"
  else
    cp "$WEB_DIR/README-DOCKER.md" "$HF_DIR/README.md"
  fi
fi

# Migrate legacy flat Gradio assets into runtime-assets/ (one-time friendly).
for asset in driver_drowsiness_model.keras mi-gente-sountec-live-edit.mp3 driver.db; do
  if [[ -f "$HF_DIR/$asset" && ! -f "$HF_DIR/runtime-assets/$asset" ]]; then
    if $DRY_RUN; then
      echo "Would move $asset → runtime-assets/"
    else
      mv "$HF_DIR/$asset" "$HF_DIR/runtime-assets/$asset"
    fi
  fi
done

if $DRY_RUN || ! $PUSH; then
  echo "Rsync finished (no git push)."
  exit 0
fi

GITHUB_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GITHUB_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

cd "$HF_DIR"
printf '{"github_sha":"%s","github_ref":"%s","synced_at":"%s","source":"sync-huggingface-space.sh"}\n' \
  "$GITHUB_SHA" "$GITHUB_BRANCH" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > .hf-sync-meta.json
git add -A
if git diff --staged --quiet; then
  echo "HF Space already matches GitHub — nothing to commit."
  exit 0
fi

git commit -m "chore: sync Docker Space from GitHub (${GITHUB_BRANCH}@${GITHUB_SHA})"
git push origin main
echo "Pushed to Hugging Face Space (main)."

#!/usr/bin/env bash
# Sync GitHub monorepo web/ → Hugging Face Space repo (flat root layout).
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

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true; PUSH=false ;;
    --no-push) PUSH=false ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$WEB_DIR" ]]; then
  echo "Missing web/: $WEB_DIR" >&2
  exit 1
fi

if [[ ! -d "$HF_DIR/.git" ]]; then
  echo "HF Space clone not found (expected git repo): $HF_DIR" >&2
  echo "Clone with: git clone https://huggingface.co/spaces/ArnavPoswal15/FocusFleet \"$HF_DIR\"" >&2
  exit 1
fi

RSYNC_FLAGS=(-av)
if $DRY_RUN; then
  RSYNC_FLAGS+=(-n --itemize-changes)
fi

# Do not use --delete: HF may hold LFS assets (e.g. *.keras) not tracked in GitHub web/.
rsync "${RSYNC_FLAGS[@]}" \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.DS_Store' \
  --exclude='venv/' \
  --exclude='*.code-workspace' \
  "$WEB_DIR/" "$HF_DIR/"

if $DRY_RUN || ! $PUSH; then
  echo "Rsync finished (no git push)."
  exit 0
fi

GITHUB_SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GITHUB_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

cd "$HF_DIR"
git add -A
if git diff --staged --quiet; then
  echo "HF Space already matches web/ — nothing to commit."
  exit 0
fi

git commit -m "chore: sync web/ from GitHub (${GITHUB_BRANCH}@${GITHUB_SHA})"
git push origin main
echo "Pushed to Hugging Face Space (main)."

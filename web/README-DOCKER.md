---
title: FocusFleet
emoji: 🌖
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
license: mit
short_description: AI-powered driver drowsiness detection (FastAPI + Next.js)
---

# FocusFleet (Docker Space)

This Hugging Face Space runs the monorepo **FastAPI backend** and **static Next.js frontend** from a single container (port **7860**).

## Layout (Space repo)

| Path | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: `frontend/` → `out/`, Python runtime with `backend/` + `focusfleet/` |
| `docker-entrypoint.sh` | Seeds runtime assets, sets `FOCUSFLEET_DATA_DIR`, starts Uvicorn |
| `runtime-assets/` | Model (`.keras`), alert `.mp3`, optional `driver.db` template (often Git LFS on HF) |
| `backend/`, `focusfleet/`, `frontend/` | Source synced from GitHub (frontend built inside Docker) |

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `FOCUSFLEET_DATA_DIR` | `/data` | Writable directory for SQLite DB, logs, and copied model/audio assets |
| `FOCUSFLEET_STATIC_DIR` | `/app/out` | Built Next.js static export (set in `Dockerfile`) |
| `FOCUSFLEET_SECRET_KEY` | random per process | Session cookie signing |
| `FOCUSFLEET_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated origins if you split frontend/API |
| `FOCUSFLEET_ADMIN_USERNAME` | _(unset)_ | Bootstrap fleet admin username (see below) |
| `FOCUSFLEET_ADMIN_PASSWORD` | _(unset)_ | Bootstrap fleet admin password (pair with username) |
| `PORT` | `7860` | Hugging Face expects the app on this port |

### Admin login on the Space

Set **`FOCUSFLEET_ADMIN_USERNAME`** and **`FOCUSFLEET_ADMIN_PASSWORD`** as [Space secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets) (Settings → Secrets). On container startup, FastAPI creates the first admin if the database has no admin row yet. Sign in at **`/login/`** on the Space URL; admins are redirected to **`/admin/`**. Use values you choose—never commit real passwords to git. After adding or changing secrets, **Factory rebuild** or restart the Space so the container picks them up. Template: [`backend/.env.example`](../backend/.env.example).

On startup, files missing under `FOCUSFLEET_DATA_DIR` are copied from `/app/runtime-assets/` (and legacy `/app/` paths) so large LFS assets can live in the Space repo without rebuilding the image layer every sync.

## Local Docker smoke test

From the GitHub monorepo root:

```bash
mkdir -p runtime-assets
cp web/mi-gente-sountec-live-edit.mp3 runtime-assets/
# optional: cp web/driver_drowsiness_model.keras web/driver.db runtime-assets/

docker build -t focusfleet .
docker run --rm -p 7860:7860 -e FOCUSFLEET_DATA_DIR=/data focusfleet
```

Large assets (`.keras`, `driver.db`) usually live in the Hugging Face Space under `runtime-assets/` via Git LFS; the entrypoint copies them into `FOCUSFLEET_DATA_DIR` on first run.

## Sync from GitHub

Pushes to `main` (or `./scripts/sync-huggingface-space.sh`) rsync `focusfleet/`, `backend/`, `frontend/`, `Dockerfile`, `docker-entrypoint.sh`, and runtime web assets into this Space. The Space rebuilds when the Docker definition or sources change.

See the root [README](https://github.com/ArnavPoswal15/FocusFleet) for full project documentation.

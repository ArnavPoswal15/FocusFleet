# FocusFleet: Driver Drowsiness Detection

FocusFleet monitors driver alertness with computer vision (MediaPipe landmarks, EAR/MAR heuristics, and an optional Keras classifier). It is aimed at fleets and safety workflows that need a clear audit trail and a path from a working demo to a self-hosted product—not hype, but a stack you can run locally and deploy incrementally.

**Product shape today:** a static **Next.js** console, a **FastAPI** backend for auth/sessions/analytics and live monitoring, and a shared **`focusfleet`** CV core. The public **Hugging Face Space** runs the same stack in a **Docker** container (`sdk: docker`, port **7860**); local Gradio under `web/` remains available for parity testing.

Diagram and layer notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Architecture

| Component | Location | Responsibility |
| --- | --- | --- |
| **Next.js (static export)** | `frontend/` | Dashboard, monitoring UI, analytics, settings. `output: "export"` for static hosting. |
| **FastAPI** | `backend/` | REST auth/sessions/analytics; WebSocket monitoring; CORS for `localhost:3000`. |
| **`focusfleet` CV library** | `focusfleet/` | Detection, state machine, SQLite helpers, Gradio UI module. |
| **HF Docker Space** | `Dockerfile`, `backend/`, `frontend/`, `runtime-assets/` | Builds Next + FastAPI image; syncs to [ArnavPoswal15/FocusFleet](https://huggingface.co/spaces/ArnavPoswal15/FocusFleet). See [`web/README-DOCKER.md`](web/README-DOCKER.md). |
| **Gradio (local / legacy)** | `web/` | Optional Gradio entry (`web/main.py`) for HF-parity CV testing. |
| **Desktop (legacy)** | `application/` | CustomTkinter app; same concepts, separate copy of engine/logger. |

**Data flow (target product path):** browser → Next static app → FastAPI (`/api/*`, `/ws/monitoring/{session_id}`) → `focusfleet.detection` / `DetectionService`.

**Data flow (Space path):** browser webcam → Gradio in `focusfleet.ui.app` → `process_frame` / `classify_image` in `focusfleet.detection`.

---

## Local development

Use separate terminals from the repo root. Python **3.10+** recommended (match HF Space: 3.10.13).

### Gradio web app (HF-parity)

Runs the same layout as the Space; good for CV/UI changes without the Next stack.

```bash
cd web
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open the URL Gradio prints (typically `http://127.0.0.1:7860`).

### Backend API

Expects the monorepo layout so `focusfleet` and `web/` (DB, model path) resolve correctly.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check: `GET http://localhost:8000/api/health`

Optional: `FOCUSFLEET_CORS_ORIGINS=http://localhost:3000` (comma-separated).

#### Fleet admin account

Admins share the same login page as drivers (`/login/`). The first admin is created on API startup when env vars are set (not via `/register`):

```bash
export FOCUSFLEET_ADMIN_USERNAME=admin
export FOCUSFLEET_ADMIN_PASSWORD='choose-a-strong-password'
cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

After sign-in, admins are redirected to `/admin/` (fleet overview, driver list, session logs). Drivers still use `/dashboard/` and cannot call `/api/admin/*`.

See [`backend/.env.example`](backend/.env.example) for a safe template (placeholders only).

#### Hugging Face Space (live app)

**Requires the Docker Space** (`sdk: docker` — see [`web/README-DOCKER.md`](web/README-DOCKER.md)). The legacy Gradio Space does not serve `/login/` or `/admin/` and does not read admin env vars on startup. Sync from GitHub with `./scripts/sync-huggingface-space.sh`, then Factory rebuild.

The Docker Space runs the same FastAPI startup hook: it reads **`FOCUSFLEET_ADMIN_USERNAME`** and **`FOCUSFLEET_ADMIN_PASSWORD`** from the container environment.

1. Open your Space → **Settings** → **Repository secrets** (or **Secrets**).
2. Add **`FOCUSFLEET_ADMIN_USERNAME`** — any username you want (e.g. `fleet-admin`).
3. Add **`FOCUSFLEET_ADMIN_PASSWORD`** — a strong password you choose (not checked into GitHub).
4. **Factory rebuild** or restart the Space so the new env vars are injected before Uvicorn starts.
5. Visit the Space URL → **`/login/`**, sign in with those credentials. Admins go to **`/admin/`**; drivers still use **`/register`** and **`/dashboard/`**.

Bootstrap runs only when SQLite has **no** admin row yet (typical on a fresh Space volume). If an admin already exists, changing secrets does not reset the password—clear or replace the DB under `FOCUSFLEET_DATA_DIR` only if you intentionally want to re-bootstrap.

More Docker env details: [`web/README-DOCKER.md`](web/README-DOCKER.md).

### Frontend (Next.js dev)

Defaults to mock data unless you point it at the API:

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). To use the real backend:

```bash
NEXT_PUBLIC_USE_MOCK=false NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Production-style static build: `npm run build` (output in `frontend/out/`).

### Desktop app (optional)

```bash
cd application
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

---

## Performance and benchmarks

Formal benchmarks are not checked in yet. Use this section as a template when you add numbers.

| Scenario | Target (TBD) | Notes |
| --- | --- | --- |
| Gradio live stream | _ms/frame @ 720p_ | `stream_every=0.08` caps callback rate (~12.5 Hz). |
| FastAPI WebSocket monitoring | _ms/frame + JSON overhead_ | Client sends JPEG frames; server runs `DetectionService`. |
| Cold start (HF Space) | _seconds_ | TensorFlow + model load on first request. |
| Static Next deploy | _First Contentful Paint_ | Static export; no SSR. |

When publishing results, record hardware, Python/TF versions, `driver_drowsiness_model.keras` loaded or not, and camera resolution.

---

## Gradio streaming hotfix and migration

**Hotfix:** In `focusfleet/ui/app.py`, the live webcam uses `webcam_input.stream(..., stream_every=0.08, concurrency_limit=2)`. Gradio 6’s streaming API otherwise invokes the frame handler too aggressively, which can stall the UI and overload inference on CPU Spaces. The status banner is decoupled via `gr.Timer(1)` so alerts do not flicker every frame.

**Migration path:**

1. **Now:** Ship CV fixes in `focusfleet/`; validate on local Gradio (`web/main.py`) and sync to HF.
2. **Next:** Drive monitoring from the Next app via FastAPI WebSocket (`backend/app/api/websocket.py`) and shared detection code—same thresholds and state machine, product UX in `frontend/`.
3. **Deploy:** Host static `frontend/out/` on any static host; run FastAPI beside it; keep the Space as a public demo or fallback until WebSocket monitoring is parity-tested.

Legacy `web/engine.py` / duplicated files under `application/` should converge on `focusfleet/` over time; treat `focusfleet` as the single source for detection behavior.

---

## Key features

- Real-time webcam monitoring with annotated frames and status states (ACTIVE → WARNING → DROWSY → FACE_LOST)
- EAR/MAR heuristics plus optional Keras validation
- Session logging on state transitions (not every frame)
- Driver registration/login (SQLite)
- Image upload classification tab (Gradio)
- Dual surface: HF Gradio Space + Next/FastAPI product shell

---

## Project structure

```
FocusFleet/
├── focusfleet/          # Shared CV, config, state, DB, Gradio UI
├── backend/             # FastAPI application
├── frontend/            # Next.js static export UI
├── web/                 # HF Space entry + requirements
├── application/         # Legacy CustomTkinter desktop app
├── docs/                # Architecture notes
└── scripts/             # HF Space sync helper
```

---

## Tech stack

- **Python:** OpenCV, MediaPipe, TensorFlow/Keras, FastAPI, Gradio
- **Frontend:** Next.js 15, React 19, Tailwind
- **Data:** SQLite (`web/driver.db` when using backend/Gradio paths)
- **Deploy:** Static Next + FastAPI on one host, or HF Space (**Docker** SDK, Python 3.10.13, `FOCUSFLEET_DATA_DIR` for SQLite/logs/model)

---

## Detection logic (summary)

1. MediaPipe Face Mesh landmarks → EAR and mouth metrics  
2. State machine with persistent thresholds → WARNING / DROWSY  
3. Optional Keras pass on cropped face regions  
4. Audio/visual alerts and session logs on transitions  

Details live in `focusfleet/detection.py`, `focusfleet/state.py`, and `focusfleet/session_logger.py`.

---

## Configuration

- **Model:** `web/driver_drowsiness_model.keras` (large; often Git LFS on HF, not always in GitHub)
- **Alert audio:** `mi-gente-sountec-live-edit.mp3` under `web/` (and desktop copy)
- **Dependencies:** `web/requirements.txt`, `backend/requirements.txt`, `application/requirements.txt`

---

## Hugging Face Space sync

**Source of truth:** `focusfleet/`, `backend/`, `frontend/` (source, not `out/`), root `Dockerfile`, `docker-entrypoint.sh`, and runtime assets (`runtime-assets/` or files under `web/` such as the alert MP3 and optional `.keras` / `driver.db`).

1. **CI:** set GitHub secret `HF_TOKEN`. Pushes to `main` that touch synced paths run `.github/workflows/sync-huggingface.yml` (Space README is copied from `web/README-DOCKER.md`).
2. **Manual:** `./scripts/sync-huggingface-space.sh` (clone at `~/Desktop/huggingface/FocusFleet` or set `HF_SPACE_DIR`).

The sync script does not use `--delete`; large LFS assets on the Space (for example `runtime-assets/driver_drowsiness_model.keras`) are kept even when absent from GitHub.

**Runtime data:** in Docker, set **`FOCUSFLEET_DATA_DIR`** (default `/data`) for writable SQLite, logs, and seeded model/audio. Details: [`web/README-DOCKER.md`](web/README-DOCKER.md).

**Live Space:** [https://huggingface.co/spaces/ArnavPoswal15/FocusFleet](https://huggingface.co/spaces/ArnavPoswal15/FocusFleet)

**Admin on live:** set Space secrets `FOCUSFLEET_ADMIN_USERNAME` and `FOCUSFLEET_ADMIN_PASSWORD`, rebuild, then log in at `/login/` (step-by-step under [Fleet admin account](#fleet-admin-account) → Hugging Face Space).

---

## License

MIT License.

---

## Support

1. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system layout  
2. `focusfleet/` and `backend/app/` for API and detection behavior  
3. Per-app READMEs under `web/`, `frontend/`, and `application/` where present  

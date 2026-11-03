# FocusFleet architecture

High-level layout of the monorepo. Commands and deploy steps: [README](../README.md). Space env vars: [web/README-DOCKER.md](../web/README-DOCKER.md).

---

## Diagrams for reading (plain text)

These render in any editor, GitHub, and the Space README—no Mermaid plugin needed.

### A. Hugging Face Space (production)

```
                         USER BROWSER
              (login, dashboard, monitoring, admin)
                              │
                              │  HTTPS · one origin
                              ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  DOCKER SPACE · port 7860                                      ┃
┃                                                                ┃
┃   ┌──────────────────┐         ┌──────────────────────────┐   ┃
┃   │ Next.js static   │  HTTP   │ FastAPI                   │   ┃
┃   │ pages + assets   │◄───────►│ REST  /api/*              │   ┃
┃   │ /app/out         │         │ WS    /ws/monitoring/…    │   ┃
┃   └──────────────────┘         └─────────────┬────────────┘   ┃
┃                                              │                 ┃
┃                                              ▼                 ┃
┃                               ┌──────────────────────────┐   ┃
┃                               │ focusfleet               │   ┃
┃                               │ face mesh · EAR · states │   ┃
┃                               │ optional Keras · alerts  │   ┃
┃                               └─────────────┬────────────┘   ┃
┃                                              │                 ┃
┃                                              ▼                 ┃
┃                               FOCUSFLEET_DATA_DIR (/data)      ┃
┃                               driver.db · logs · model · mp3   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Auth: POST /api/auth/login → access_token (browser storage + Authorization header)
```

### B. Local development

```
  Browser ──► Next.js dev server (:3000)
                    │
                    │  /api/* proxied in dev
                    ▼
              FastAPI (:8000)
                    │
                    ├──► focusfleet/  (detection code)
                    └──► web/         (DB + model paths on disk)
```

### C. Code → Hugging Face

```
  GitHub (main)  ──push──►  GitHub Actions + HF_TOKEN
                                    │
                                    ▼
                          HF Space git repository
                                    │
                                    ▼
                          Docker build → Running app
```

### D. Monitoring session (driver)

```
  1. Login (token stored)
  2. POST /api/sessions          → start log + tracker
  3. WebSocket: send camera JPEG → detect → JSON + annotated frame back
  4. POST /api/sessions/{id}/stop → finalize session_log.txt
  5. Dashboard / analytics read logs under /data/logs/{name}/
```

---

## Mermaid versions (GitHub preview)

### 1. Production (Hugging Face Docker Space)

One container on port **7860**: static Next.js UI and FastAPI share the same origin.

```mermaid
flowchart TB
  subgraph user["Browser"]
    UI["Next.js static pages<br/>/login · /dashboard · /monitoring · /admin"]
  end

  subgraph space["HF Docker Space port 7860"]
    Uvicorn["Uvicorn · FastAPI"]
    Static["Static files /app/out"]
    API["REST /api/*"]
    WS["WebSocket /ws/monitoring/session_id"]
    CV["focusfleet + DetectionService"]
    Data["/data · SQLite · logs · model · audio"]
    Uvicorn --> Static
    Uvicorn --> API
    Uvicorn --> WS
    API --> CV
    WS --> CV
    API --> Data
    CV --> Data
  end

  UI -->|"same origin · Bearer token"| API
  UI -->|"WebSocket and access_token"| WS
  UI --> Static
```

| Piece | Path / note |
| --- | --- |
| Image build | `Dockerfile`: `npm run build` → `frontend/out` → `/app/out` |
| Start | `docker-entrypoint.sh`: seed assets → `FOCUSFLEET_DATA_DIR=/data` → Uvicorn |
| Auth | Login returns `access_token` (localStorage + `Authorization` header); cookies optional |
| Admin | Space secrets `FOCUSFLEET_ADMIN_*` → upsert admin user on startup |

---

## 2. Local development

Two processes (or three with Gradio-only CV testing):

```mermaid
flowchart LR
  subgraph dev["Local machine"]
    NextDev["Next dev :3000<br/>proxies /api → :8000"]
    API["FastAPI :8000"]
    FF["focusfleet/"]
    WebData["web/ · driver.db · model paths"]
    NextDev --> API
    API --> FF
    API --> WebData
    FF --> WebData
  end

  Browser["Browser"] --> NextDev
```

Optional: `web/main.py` (Gradio) talks to `focusfleet/` directly without the Next stack.

---

## 3. GitHub → Hugging Face sync

```mermaid
flowchart LR
  GH["GitHub main"]
  Action["GitHub Action<br/>HF_TOKEN"]
  SpaceRepo["HF Space git repo"]
  Build["Docker build on HF"]
  Run["Running Space"]

  GH -->|"push synced paths"| Action
  Action -->|"rsync + commit"| SpaceRepo
  SpaceRepo --> Build --> Run
```

Synced: `focusfleet/`, `backend/`, `frontend/`, `Dockerfile`, `docker-entrypoint.sh`, runtime assets.  
Not synced: local logs, `.venv`, `node_modules`, `.env.local`.

---

## 4. Live monitoring (product path)

```mermaid
sequenceDiagram
  participant B as Browser
  participant N as Next UI
  participant A as FastAPI
  participant W as WebSocket
  participant D as focusfleet detection

  B->>N: Login
  N->>A: POST /api/auth/login
  A-->>N: access_token
  B->>N: Start monitoring
  N->>A: POST /api/sessions
  A->>D: DriverStateTracker and session log
  loop Each camera frame
    B->>W: JPEG bytes
    W->>D: process_frame
    D-->>W: status and annotated JPEG
    W-->>B: JSON and frame_b64
  end
  N->>A: POST /api/sessions/(session_id)/stop
```

---

## 5. Detection pipeline (shared)

```mermaid
flowchart TD
  Frame["Camera frame"]
  MP["MediaPipe Face Mesh"]
  EAR["EAR + mouth / yawn"]
  SM["State machine<br/>ACTIVE to WARNING to DROWSY"]
  K["Optional Keras classifier"]
  Alert["Alert audio · session logs"]
  Frame --> MP --> EAR --> SM
  EAR --> K
  SM --> Alert
```

Implementation: `focusfleet/detection.py`, `focusfleet/state.py`, `focusfleet/session_logger.py`, `backend/app/services/detection_service.py`.

---

## Repo map

| Directory | Role |
| --- | --- |
| `frontend/` | Next.js static UI |
| `backend/` | FastAPI app (`app/main.py`, routers, WebSocket) |
| `focusfleet/` | Shared CV, config, session logging |
| `web/` | Gradio entry for local CV tests; source for Space README frontmatter |
| `application/` | Legacy desktop app |
| `scripts/` | `sync-huggingface-space.sh` |

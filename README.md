# FocusFleet: Driver Drowsiness Detection System 🚗💤

FocusFleet is a production-ready AI-powered Driver Drowsiness Detection application designed to enhance road safety. It utilizes computer vision and machine learning to monitor driver attentiveness in real-time, providing both visual and audible alerts when signs of fatigue or distraction are detected.

---

## 🌟 Key Features

- **Real-time Monitoring**: High-performance video stream processing with non-blocking UI
- **Intelligent Detection**: Uses Eye Aspect Ratio (EAR) and mouth gap analysis via MediaPipe Face Mesh
- **Custom ML Model**: Integrated TensorFlow/Keras model for secondary validation of driver state
- **State-Based Logging**: Event-driven logging system tracking sessions and state transitions
- **Audible Alerts**: Automatic sound alerts when drowsy state is confirmed
- **User Management**: Built-in SQLite database for driver registration and login
- **Dual Deployment**: Desktop app (CustomTkinter) + Web app (Gradio/Hugging Face)
- **Image Classification**: Manual image testing and classification capabilities

---

## 📂 Project Structure & Location Guide

```
FocusFleet/
├── 📁 application/           # DESKTOP APPLICATION (CustomTkinter)
│   ├── 🐍 app.py             # Main desktop GUI application
│   ├── 🧠 engine.py          # ML inference engine (MediaPipe + Keras)
│   ├── 📊 logger.py          # State management and session logger
│   ├── 🎵 mi-gente-sountec-live-edit.mp3  # Alert sound file
│   ├── 💾 driver.db          # SQLite database for user management
│   ├── 📋 requirements.txt   # Python dependencies for desktop app
│   ├── 📋 requirements_hf.txt # Hugging Face specific dependencies
│   ├── 🛠️ build_script.py    # Build and deployment script
│   ├── 🖥️ gradio.py          # Gradio interface (alternative desktop web)
│   └── 📄 README.md          # Desktop app specific documentation
│
├── 📁 web/                   # WEB APPLICATION (Gradio/Hugging Face)
│   ├── 🌐 app.py             # Gradio web application for cloud deployment
│   ├── 🧠 engine.py          # Shared ML inference engine
│   ├── 📊 logger.py          # Shared state management
│   ├── 🎵 mi-gente-sountec-live-edit.mp3  # Alert sound file
│   ├── 💾 driver.db          # Database for web app
│   ├── 🤖 driver_drowsiness_model.keras  # Pre-trained Keras model
│   ├── 📋 requirements.txt   # Web app dependencies
│   ├── ⚙️ .gitattributes      # Git configuration for large files
│   ├── 🚫 .gitignore         # Git ignore file
│   └── 📄 README.md          # Web app specific documentation
│
└── 📄 README.md              # THIS FILE - Combined project documentation
```

---

## 🛠️ Tech Stack

### Core Technologies
- **Python**: 3.10+ (primary language)
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, MediaPipe Face Mesh
- **Database**: SQLite for user management
- **Audio**: Native OS calls (`afplay` for macOS, `winsound` for Windows)

### Desktop Application (`/application/`)
- **GUI Framework**: CustomTkinter
- **Interface**: Native desktop application
- **Deployment**: Local installation

### Web Application (`/web/`)
- **Web Framework**: Gradio
- **Deployment**: Hugging Face Spaces
- **Interface**: Browser-based web application

---

## 🚀 Getting Started

### Option 1: Desktop Application (Local)

Navigate to the `application/` directory:

```bash
cd application
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Option 2: Web Application (Cloud)

The web version is optimized for Hugging Face Spaces deployment:

- **Live URL**: [https://huggingface.co/spaces/ArnavPoswal15/FocusFleet](https://huggingface.co/spaces/ArnavPoswal15/FocusFleet)
- **Local Testing**: Navigate to `web/` directory and run:
  ```bash
  cd web
  pip install -r requirements.txt
  python app.py
  ```

---

## 📊 Detection Logic & Architecture

### Core Components Location

1. **ML Engine** (`engine.py` - both directories)
   - MediaPipe Face Mesh integration (468 facial landmarks)
   - EAR (Eye Aspect Ratio) calculation
   - MAR (Mouth Aspect Ratio) for yawn detection
   - Keras model inference for validation

2. **State Management** (`logger.py` - both directories)
   - Session tracking and logging
   - State machine: ACTIVE → WARNING → DROWSY → FACE_LOST
   - Event-driven logging (not per-frame to avoid bloat)

3. **User Interface**
   - Desktop: `application/app.py` (CustomTkinter GUI)
   - Web: `web/app.py` (Gradio interface)

### Detection Workflow

1. **Face Detection**: MediaPipe extracts 468 facial landmarks
2. **Metric Calculation**:
   - EAR: Monitors eye openness/closure
   - MAR: Detects yawning patterns
3. **State Machine**: Transitions based on persistent threshold violations
4. **Alert System**: Triggers audio alerts when DROWSY state persists
5. **Logging**: Records state changes and session data

---

## 🔧 Configuration & Customization

### Model Files
- **Keras Model**: `web/driver_drowsiness_model.keras` (797MB)
- **Alert Sound**: `mi-gente-sountec-live-edit.mp3` (both directories)

### Database Schema
Both applications use SQLite with the same schema:
```sql
CREATE TABLE Driver (
    DriverID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Password TEXT NOT NULL,
    ContactInfo TEXT,
    LicenseNumber TEXT
)
```

### Dependencies
- **Desktop**: See `application/requirements.txt`
- **Web**: See `web/requirements.txt`

---

## 🌐 Deployment Information

### Desktop Deployment
- **Target**: Local machines (Windows, macOS, Linux)
- **Package**: Standalone Python application
- **Dependencies**: Managed via virtual environment

### Web Deployment
- **Platform**: Hugging Face Spaces
- **Python Version**: 3.10.13
- **SDK**: Gradio 6.8.0
- **Auto-rebuild**: Triggered on main branch pushes

### Keeping GitHub and Hugging Face in sync

**Source of truth:** edit the web app under `web/` in this GitHub repo only. The Space repo is a flat copy of that folder (not the whole monorepo).

1. **Automatic (recommended):** add a GitHub repository secret `HF_TOKEN` ([Hugging Face write token](https://huggingface.co/settings/tokens)). Pushes to `main` that touch `web/` run `.github/workflows/sync-huggingface.yml` and update the Space.
2. **Manual:** from the repo root, run `./scripts/sync-huggingface-space.sh` (requires a local clone at `~/Desktop/huggingface/FocusFleet`, or set `HF_SPACE_DIR`).

The sync script does **not** delete large files on the Space (for example `driver_drowsiness_model.keras`) if they are not in GitHub; keep the model on HF via Git LFS there.

---

## 🤝 Development & Contributing

### Shared Components
- `engine.py` and `logger.py` are shared between both applications
- Core detection logic remains consistent across platforms
- Database schema is identical for data compatibility

### Testing
- Desktop: Run `application/app.py` locally
- Web: Test via Hugging Face Spaces or local Gradio server

---

## 📄 License

This project is licensed under the MIT License.

---

## 📞 Support

For issues or questions:
1. Check the respective README files in `application/` or `web/` directories
2. Review the detection logic in `engine.py` and `logger.py`
3. Test with the appropriate interface (desktop vs web)

import gradio as gr
import os
import sqlite3
import datetime
import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# GLOBAL VARIABLES FOR LOGGING & ALERTS

LOG_FILE = None
SESSION_ID = None

import sys
import threading

def play_alert_sound():
    def _play():
        sound_path = "mi-gente-sountec-live-edit.mp3"
        try:
            if sys.platform == "darwin":
                os.system(f'afplay "{sound_path}"')
            else:
                import winsound
                winsound.Beep(1000, 500)
        except Exception:
            print("ALERT SOUND: Drowsiness detected!")
            
    threading.Thread(target=_play, daemon=True).start()


# Initialize tracker globally
TRACKER = None

# ── Status state written by the video stream, read by the Timer at 1 Hz ──
# This completely decouples the fast video loop from the slow banner update.
DROWSY_SINCE: float | None = None
DROWSY_CONFIRM_SECS = 2.0
CURRENT_STATUS_KEY: str = "neutral"   # alert | warning | drowsy | no_face | neutral

from logger import DriverStateTracker

def write_log(message):
    global LOG_FILE
    if LOG_FILE:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"[{timestamp}] {message}")

# DATABASE SETUP

DB_NAME = "driver.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS Driver (
            DriverID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name TEXT NOT NULL,
            Password TEXT NOT NULL,
            ContactInfo TEXT,
            LicenseNumber TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------------------
# LOGIN & REGISTRATION FUNCTIONS
# ---------------------
def login_driver(name, password):
    if name.strip() == "" or password.strip() == "":
        return None, "Please enter both name and password."
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM Driver WHERE Name=? AND Password=?", (name, password))
    row = c.fetchone()
    conn.close()
    if row:
        driver_info = {"DriverID": row[0], "Name": row[1], "ContactInfo": row[3], "LicenseNumber": row[4]}
        # Set up logging: create a logs folder under driver name with a session timestamp
        session_start = datetime.datetime.now()
        session_id = session_start.strftime("%Y%m%d_%H%M%S")
        logs_dir = os.path.join("logs", driver_info["Name"], session_id)
        os.makedirs(logs_dir, exist_ok=True)
        global LOG_FILE, SESSION_ID, TRACKER
        LOG_FILE = os.path.join(logs_dir, "session_log.txt")
        SESSION_ID = session_id
        
        TRACKER = DriverStateTracker(LOG_FILE)
        
        return driver_info, f"Welcome, {row[1]}!"
    else:
        return None, "Invalid credentials."

def register_driver(name, password, contact, license_num):
    if name.strip() == "" or password.strip() == "":
        return "Name and Password are required."
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO Driver (Name, Password, ContactInfo, LicenseNumber) VALUES (?, ?, ?, ?)",
              (name, password, contact, license_num))
    conn.commit()
    conn.close()
    return "Registration successful! You can now log in."

# DROWSINESS DETECTION SETUP (Using MediaPipe Face Mesh for EAR and mouth detection)
model = tf.keras.models.load_model("driver_drowsiness_model.keras")

haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_EAR(landmarks, eye_indices, image_width, image_height):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * image_width), int(lm.y * image_height)))
    A = euclidean_distance(pts[1], pts[5])
    B = euclidean_distance(pts[2], pts[4])
    C = euclidean_distance(pts[0], pts[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness(frame, ear_threshold):
    """
    Process a live frame (RGB numpy array) by:
      - Resizing to 640x480 for better display.
      - Running MediaPipe Face Mesh.
      - Computing EAR for both eyes.
      - Computing mouth opening (as a proxy for yawning) using landmarks 13 and 14.
      - If EAR is below threshold OR mouth opening exceeds threshold, label as Drowsy.
      - Draw a bounding box around the face (computed from landmarks) with a red color if drowsy, green otherwise.
      - Show drowsy and not-drowsy percentages.
    Returns (frame, status_string) for UI status banner.
    """
    if frame is None:
        return None, "No camera feed"
    # Resize frame to 640x480 for display
    frame_disp = cv2.resize(frame, (640, 480))
    h, w, _ = frame_disp.shape

    # Process frame with MediaPipe Face Mesh
    results = face_mesh.process(frame_disp)
    if not results.multi_face_landmarks:
        cv2.putText(frame_disp, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        if TRACKER:
            TRACKER.update_state_v2(ear_threshold, face_detected=False, is_yawning=False, metrics={})
        return frame_disp, "No face detected"

    landmarks = results.multi_face_landmarks[0].landmark

    left_ear = compute_EAR(landmarks, LEFT_EYE_IDX, w, h)
    right_ear = compute_EAR(landmarks, RIGHT_EYE_IDX, w, h)
    avg_ear = (left_ear + right_ear) / 2.0

    lm_top = landmarks[MOUTH_TOP_IDX]
    lm_bottom = landmarks[MOUTH_BOTTOM_IDX]
    mouth_distance = euclidean_distance((lm_top.x * w, lm_top.y * h),
                                        (lm_bottom.x * w, lm_bottom.y * h))
    # Define mouth threshold (adjustable; here relative to frame height)
    mouth_threshold = 0.08 * h  # e.g., 8% of frame height

    # Determine if drowsy: if EAR is below threshold OR mouth opening is large
    drowsy_flag = (avg_ear < ear_threshold) or (mouth_distance > mouth_threshold)


    # drowsy_pct_mouth = min(100, ((mouth_distance - mouth_threshold) / mouth_threshold)*100)
    drowsy_pct_ear = max(0, min(100, (1 - avg_ear/ear_threshold)*100)) if avg_ear < ear_threshold else 0
    drowsy_pct_mouth = max(0, min(100, ((mouth_distance - mouth_threshold) / mouth_threshold)*100)) if mouth_distance > mouth_threshold else 0
    # Combine percentages (take maximum as a simple rule)
    drowsy_pct = max(drowsy_pct_ear, drowsy_pct_mouth)
    not_drowsy_pct = 100 - drowsy_pct

    # Compute face bounding box from landmarks
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min = int(min(xs) * w)
    y_min = int(min(ys) * h)
    x_max = int(max(xs) * w)
    y_max = int(max(ys) * h)

    # Update State Tracker
    if TRACKER:
        is_yawning = (mouth_distance > mouth_threshold)
        metrics = {"avg_ear": avg_ear, "mouth_dist": mouth_distance}
        TRACKER.update_state_v2(ear_threshold, face_detected=True, is_yawning=is_yawning, metrics=metrics)
        
        # Check Tracker state for alarm
        if TRACKER.current_state.value == "DROWSY":
            play_alert_sound()
            
    # Set label and color
    if drowsy_flag:
        label_text = f"Drowsy ({drowsy_pct:.1f}% drowsy, {not_drowsy_pct:.1f}% alert)"
        color = (0, 0, 255)  # red
    else:
        label_text = f"Not Drowsy ({not_drowsy_pct:.1f}% alert, {drowsy_pct:.1f}% drowsy)"
        color = (0, 255, 0)  # green

    # Draw the bounding box (without drawing individual landmarks)
    cv2.rectangle(frame_disp, (x_min, y_min), (x_max, y_max), color, 3)
    cv2.putText(frame_disp, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2, cv2.LINE_AA)

    status = "⚠️ DROWSINESS DETECTED — Stay alert!" if drowsy_flag else "✓ Alert"
    return frame_disp, status


# ── Static HTML constants for each state ─────────────────────────────────────
STATUS_HTML = {
    "alert":   '<p class="status-banner status-alert">✓ Alert — Driver is focused</p>',
    "warning": '<p class="status-banner status-warning">⚠ Monitoring — keep your eyes on the road</p>',
    "drowsy":  '<p class="status-banner status-drowsy status-drowsy-blink">⚠️ DROWSINESS DETECTED — Stay alert!</p>',
    "no_face": '<p class="status-banner status-neutral">No face detected</p>',
    "neutral": '<p class="status-banner status-neutral">—</p>',
}


def detect_drowsiness_stream(frame, ear_threshold):
    """Video-only stream handler.

    Runs every frame at full webcam speed. Writes the current state key into
    CURRENT_STATUS_KEY (a single string). Does NOT touch the status banner —
    that is the Timer's job, running at 1 Hz. This keeps the video loop fast
    and the banner completely flicker-free.
    """
    global DROWSY_SINCE, CURRENT_STATUS_KEY

    frame_out, status_str = detect_drowsiness(frame, ear_threshold)

    now = time.time()
    raw_drowsy = "DROWSINESS" in status_str or "Drowsy" in status_str

    if raw_drowsy:
        if DROWSY_SINCE is None:
            DROWSY_SINCE = now
        CURRENT_STATUS_KEY = "drowsy" if (now - DROWSY_SINCE) >= DROWSY_CONFIRM_SECS else "warning"
    else:
        DROWSY_SINCE = None
        CURRENT_STATUS_KEY = "no_face" if ("No face" in status_str or "camera" in status_str.lower()) else "alert"

    return frame_out


def poll_status():
    """Called by gr.Timer at 1 Hz. Returns the current banner HTML.

    Because this fires at most once per second (not per frame), the DOM
    update is infrequent and deliberate — no flicker, no blink on stable states.
    """
    return STATUS_HTML.get(CURRENT_STATUS_KEY, STATUS_HTML["neutral"])

# For static image upload classification (using Haar Cascade & TensorFlow model)
def classify_image(frame):
    if frame is None:
        return "No image provided."
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return "No face detected."
    for (x, y, w_box, h_box) in faces:
        face_roi = frame_bgr[y:y+h_box, x:x+w_box]
        try:
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
        except Exception as e:
            continue
        face_normalized = face_resized.astype("float32") / 255.0
        face_array = np.expand_dims(face_normalized, axis=0)
        prediction = model.predict(face_array)
        label_index = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][label_index] * 100
        label = "Drowsy" if label_index == 0 else "Not Drowsy"
        return f"{label}: {confidence:.2f}%"
    return "No face detected."

# BUILD THE GRADIO APP

CUSTOM_CSS = """
/* ===== GLOBAL ===== */
*, *::before, *::after { box-sizing: border-box; }

/* ===== HEADER — force full-width centered block ===== */
#ff-header {
    width: 100% !important;
    text-align: center !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 2rem 1rem 1.25rem !important;
    border-bottom: 1px solid rgba(45, 212, 191, 0.15) !important;
    margin-bottom: 0.5rem !important;
    background: linear-gradient(180deg, rgba(13, 148, 136, 0.07) 0%, transparent 100%) !important;
}
#ff-header .ff-logo-line {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
}
#ff-header .ff-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #0d9488;
    box-shadow: 0 0 8px #0d9488;
    flex-shrink: 0;
}
#ff-header h1 {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    background: linear-gradient(135deg, #2dd4bf 0%, #0d9488 60%, #134e4a 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.03em;
}
#ff-header .ff-sub {
    color: #64748b;
    font-size: 0.88rem;
    margin: 0;
    letter-spacing: 0.02em;
}

/* ===== STATUS BANNER — dark-mode safe ===== */
@keyframes drowsy-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 6px rgba(239, 68, 68, 0.2); }
}
.status-banner {
    padding: 0.9rem 1.5rem;
    border-radius: 10px;
    font-weight: 700;
    text-align: center;
    font-size: 1rem;
    width: 100%;
    margin: 0.25rem 0;
    letter-spacing: 0.01em;
    transition: background 0.4s ease, border-color 0.4s ease, color 0.4s ease;
}
.status-drowsy {
    background: rgba(220, 38, 38, 0.18) !important;
    color: #fca5a5 !important;
    border: 1.5px solid rgba(239, 68, 68, 0.4) !important;
}
/* Blink only applied after 2s persistence (class added by Python) */
.status-drowsy-blink {
    animation: drowsy-pulse 1.2s ease-in-out infinite !important;
}
.status-alert {
    background: rgba(16, 185, 129, 0.15) !important;
    color: #6ee7b7 !important;
    border: 1.5px solid rgba(16, 185, 129, 0.35) !important;
    animation: none !important;
}
.status-warning {
    background: rgba(234, 179, 8, 0.12) !important;
    color: #fde68a !important;
    border: 1.5px solid rgba(234, 179, 8, 0.3) !important;
    animation: none !important;
}
.status-neutral {
    background: rgba(71, 85, 105, 0.25) !important;
    color: #94a3b8 !important;
    border: 1.5px solid rgba(100, 116, 139, 0.3) !important;
    animation: none !important;
}

/* ===== EAR SLIDER — remove side padding / expand full width ===== */
#ear-slider {
    width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}
#ear-slider > div {
    padding: 0.75rem 0 !important;
}

/* ===== STATUS HTML wrapper — fix white-box Gradio bug ===== */
#status-html {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
#status-html > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ===== CAMERA ROW ===== */
#monitor-row {
    gap: 1rem !important;
}

/* ===== WELCOME BANNER ===== */
#welcome-msg {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    background: rgba(45, 212, 191, 0.08);
    border: 1px solid rgba(45, 212, 191, 0.15);
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    color: #94a3b8;
}

/* ===== AUTH TABS styling ===== */
.auth-section-label {
    font-size: 1rem;
    font-weight: 700;
    color: #2dd4bf;
    margin-bottom: 0.5rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    font-size: 0.78rem;
}

/* ===== UPLOAD TAB ===== */
#upload-result textarea {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}

/* ===== Gradio footer cleanup ===== */
footer { opacity: 0.5 !important; }
"""

HEADER_HTML = """
<div id="ff-header">
  <div class="ff-logo-line">
    <span class="ff-dot"></span>
    <h1>FocusFleet</h1>
  </div>
  <p class="ff-sub">AI-powered driver drowsiness detection &mdash; stay safe on the road</p>
</div>
"""

with gr.Blocks(
    title="FocusFleet — Driver Drowsiness Detection",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate"),
) as demo:
    driver_state = gr.State(value=None)

    gr.HTML(HEADER_HTML)

    with gr.Tab("Login / Register", id="auth"):
        gr.Markdown("Sign in or create an account to start monitoring.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<p class="auth-section-label">Login</p>')
                login_name = gr.Textbox(label="Name", placeholder="Your name")
                login_password = gr.Textbox(label="Password", type="password", placeholder="••••••••")
                login_button = gr.Button("Login", variant="primary")
                login_status = gr.Textbox(label="Status", interactive=False, visible=True)
            with gr.Column(scale=1):
                gr.HTML('<p class="auth-section-label">Register</p>')
                reg_name = gr.Textbox(label="Name", placeholder="Your name")
                reg_password = gr.Textbox(label="Password", type="password", placeholder="••••••••")
                reg_contact = gr.Textbox(label="Contact Info", placeholder="Email or phone (optional)")
                reg_license = gr.Textbox(label="License Number", placeholder="Optional")
                register_button = gr.Button("Register", variant="secondary")
                reg_status = gr.Textbox(label="Status", interactive=False, visible=True)

        def login_callback(name, password):
            driver_info, msg = login_driver(name, password)
            return driver_info, msg

        login_button.click(login_callback, inputs=[login_name, login_password],
                           outputs=[driver_state, login_status])

        def register_callback(name, password, contact, license_num):
            return register_driver(name, password, contact, license_num)

        register_button.click(register_callback, inputs=[reg_name, reg_password, reg_contact, reg_license],
                              outputs=reg_status)

    with gr.Tab("Live Monitoring", id="monitor"):
        welcome_msg = gr.Markdown(
            value="Please log in to access live drowsiness detection.",
            elem_id="welcome-msg",
        )
        ear_threshold_slider = gr.Slider(
            minimum=0.1, maximum=0.5, value=0.25, step=0.01,
            label="EAR threshold (lower = more sensitive)",
            elem_id="ear-slider",
        )
        with gr.Row(elem_id="monitor-row"):
            webcam_input = gr.Image(
                sources=["webcam"], type="numpy", streaming=True,
                label="Webcam", height=380, show_label=True,
            )
            detection_output = gr.Image(
                label="Detection", interactive=False, height=380, show_label=True,
            )
        status_banner = gr.HTML(
            value=STATUS_HTML["neutral"],
            elem_id="status-html",
        )

        # Stream: video frames only — no banner output, runs at full camera FPS
        webcam_input.stream(
            fn=detect_drowsiness_stream,
            inputs=[webcam_input, ear_threshold_slider],
            outputs=[detection_output],
        )

        # Timer: banner only — fires at 1 Hz, completely independent of frame rate
        status_timer = gr.Timer(value=1)
        status_timer.tick(fn=poll_status, outputs=[status_banner])

    with gr.Tab("Image Check", id="upload"):
        gr.Markdown("Upload a single image to classify drowsiness (no login required).")
        upload_input = gr.Image(type="numpy", label="Upload Image", height=340)
        upload_prediction = gr.Textbox(
            label="Result", interactive=False, elem_id="upload-result"
        )
        upload_input.change(fn=classify_image, inputs=upload_input, outputs=upload_prediction)

    def update_welcome(driver_info):
        if driver_info is not None:
            return f"**Welcome, {driver_info['Name']}!** Use the webcam below to monitor your alertness."
        return "Please log in to access live drowsiness detection."

    driver_state.change(fn=update_welcome, inputs=driver_state, outputs=welcome_msg)

demo.launch()

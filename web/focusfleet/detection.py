"""Computer-vision drowsiness detection.

Two entry points are used by the UI layer:
  - process_frame(frame, ear_threshold) -> (annotated_frame, status_key)
    used by the live webcam stream.
  - classify_image(frame) -> str
    used by the static image-upload tab.
"""

import os
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from . import config
from .state import STATE

# ── Model / cascade / face mesh setup (loaded once at import time) ─────
_model = None
if os.path.exists(config.MODEL_PATH):
    try:
        _model = tf.keras.models.load_model(config.MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load model from {config.MODEL_PATH}: {e}")

_haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_haar_cascade_path)

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def play_alert_sound() -> None:
    """Fire-and-forget audible alert; falls back to a console message."""

    def _play():
        try:
            if sys.platform == "darwin":
                os.system(f'afplay "{config.ALERT_SOUND_PATH}"')
            else:
                import winsound
                winsound.Beep(1000, 500)
        except Exception:
            print("ALERT SOUND: Drowsiness detected!")

    threading.Thread(target=_play, daemon=True).start()


def _euclidean(p1, p2) -> float:
    return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)


def _compute_ear(landmarks, eye_indices, width, height) -> float:
    pts = [
        (int(landmarks[i].x * width), int(landmarks[i].y * height))
        for i in eye_indices
    ]
    vertical_a = _euclidean(pts[1], pts[5])
    vertical_b = _euclidean(pts[2], pts[4])
    horizontal = _euclidean(pts[0], pts[3])
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def detect_drowsiness(frame, ear_threshold: float):
    """Run face-mesh based EAR/yawn detection on a single frame.

    Draws a bounding box + label on the frame and updates the shared
    DriverStateTracker (if a session is active). Returns (frame, status_str).
    """
    if frame is None:
        return None, "No camera feed"

    frame_disp = cv2.resize(frame, config.FRAME_DISPLAY_SIZE)
    h, w, _ = frame_disp.shape

    results = _face_mesh.process(frame_disp)
    if not results.multi_face_landmarks:
        cv2.putText(frame_disp, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if STATE.tracker:
            STATE.tracker.update_state_v2(
                ear_threshold, face_detected=False, is_yawning=False, metrics={}
            )
        return frame_disp, "No face detected"

    landmarks = results.multi_face_landmarks[0].landmark

    left_ear = _compute_ear(landmarks, config.LEFT_EYE_IDX, w, h)
    right_ear = _compute_ear(landmarks, config.RIGHT_EYE_IDX, w, h)
    avg_ear = (left_ear + right_ear) / 2.0

    top = landmarks[config.MOUTH_TOP_IDX]
    bottom = landmarks[config.MOUTH_BOTTOM_IDX]
    mouth_distance = _euclidean((top.x * w, top.y * h), (bottom.x * w, bottom.y * h))
    mouth_threshold = config.MOUTH_THRESHOLD_RATIO * h

    is_yawning = mouth_distance > mouth_threshold
    drowsy_flag = (avg_ear < ear_threshold) or is_yawning

    drowsy_pct_ear = (
        max(0, min(100, (1 - avg_ear / ear_threshold) * 100))
        if avg_ear < ear_threshold else 0
    )
    drowsy_pct_mouth = (
        max(0, min(100, ((mouth_distance - mouth_threshold) / mouth_threshold) * 100))
        if is_yawning else 0
    )
    drowsy_pct = max(drowsy_pct_ear, drowsy_pct_mouth)
    not_drowsy_pct = 100 - drowsy_pct

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, y_min = int(min(xs) * w), int(min(ys) * h)
    x_max, y_max = int(max(xs) * w), int(max(ys) * h)

    if STATE.tracker:
        STATE.tracker.update_state_v2(
            ear_threshold,
            face_detected=True,
            is_yawning=is_yawning,
            metrics={"avg_ear": avg_ear, "mouth_dist": mouth_distance},
        )
        if STATE.tracker.current_state.value == "DROWSY":
            play_alert_sound()

    if drowsy_flag:
        label = f"Drowsy ({drowsy_pct:.1f}% drowsy, {not_drowsy_pct:.1f}% alert)"
        color = (0, 0, 255)
    else:
        label = f"Not Drowsy ({not_drowsy_pct:.1f}% alert, {drowsy_pct:.1f}% drowsy)"
        color = (0, 255, 0)

    cv2.rectangle(frame_disp, (x_min, y_min), (x_max, y_max), color, 3)
    cv2.putText(frame_disp, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    status = "⚠️ DROWSINESS DETECTED — Stay alert!" if drowsy_flag else "✓ Alert"
    return frame_disp, status


def process_frame(frame, ear_threshold: float):
    """Stream handler used by the UI: runs detection and updates the
    debounced status key in shared state. Returns the annotated frame only
    (the status banner is read separately by a slower poller)."""
    frame_out, status_str = detect_drowsiness(frame, ear_threshold)

    now = time.time()
    raw_drowsy = "DROWSINESS" in status_str or "Drowsy" in status_str

    if raw_drowsy:
        if STATE.drowsy_since is None:
            STATE.drowsy_since = now
        sustained = (now - STATE.drowsy_since) >= config.DROWSY_CONFIRM_SECS
        STATE.current_status_key = config.STATUS_DROWSY if sustained else config.STATUS_WARNING
    else:
        STATE.drowsy_since = None
        no_face = "No face" in status_str or "camera" in status_str.lower()
        STATE.current_status_key = config.STATUS_NO_FACE if no_face else config.STATUS_ALERT

    return frame_out


def classify_image(frame) -> str:
    """Static-image drowsiness classifier used by the upload tab."""
    if frame is None:
        return "No image provided."

    if _model is None:
        return "Model file not loaded (driver_drowsiness_model.keras not found)."

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return "No face detected."

    for (x, y, fw, fh) in faces:
        face_roi = frame_bgr[y:y + fh, x:x + fw]
        try:
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
        except Exception:
            continue

        face_array = np.expand_dims(face_resized.astype("float32") / 255.0, axis=0)
        prediction = _model.predict(face_array)
        label_index = int(np.argmax(prediction, axis=1)[0])
        confidence = prediction[0][label_index] * 100
        label = "Drowsy" if label_index == 0 else "Not Drowsy"
        return f"{label}: {confidence:.2f}%"

    return "No face detected."

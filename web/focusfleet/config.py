"""
Central configuration for FocusFleet.

Keeping every constant here means detection tuning, DB location, and UI
copy can all be changed in one place without touching logic elsewhere.
"""

# ── Database ────────────────────────────────────────────────────────────
DB_NAME = "driver.db"

# ── Model / cascade paths ──────────────────────────────────────────────
MODEL_PATH = "driver_drowsiness_model.keras"
ALERT_SOUND_PATH = "mi-gente-sountec-live-edit.mp3"

# ── MediaPipe Face Mesh landmark indices ───────────────────────────────
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14

# ── Detection tuning ────────────────────────────────────────────────────
DEFAULT_EAR_THRESHOLD = 0.25
EAR_THRESHOLD_MIN = 0.10
EAR_THRESHOLD_MAX = 0.50
EAR_THRESHOLD_STEP = 0.01

MOUTH_THRESHOLD_RATIO = 0.08  # fraction of frame height counted as a yawn
DROWSY_CONFIRM_SECS = 2.0     # seconds of sustained drowsy signal before alarm
FRAME_DISPLAY_SIZE = (640, 480)

# ── Status keys used across detection + UI ─────────────────────────────
STATUS_ALERT = "alert"
STATUS_WARNING = "warning"
STATUS_DROWSY = "drowsy"
STATUS_NO_FACE = "no_face"
STATUS_NEUTRAL = "neutral"

STATUS_HTML = {
    STATUS_ALERT: (
        '<div class="status-banner status-alert">'
        '<span class="status-dot"></span>Alert — Driver is focused</div>'
    ),
    STATUS_WARNING: (
        '<div class="status-banner status-warning">'
        '<span class="status-dot"></span>Monitoring — keep your eyes on the road</div>'
    ),
    STATUS_DROWSY: (
        '<div class="status-banner status-drowsy status-drowsy-blink">'
        '<span class="status-dot"></span>DROWSINESS DETECTED — Stay alert!</div>'
    ),
    STATUS_NO_FACE: (
        '<div class="status-banner status-neutral">'
        '<span class="status-dot"></span>No face detected</div>'
    ),
    STATUS_NEUTRAL: (
        '<div class="status-banner status-neutral">'
        '<span class="status-dot"></span>Waiting for input…</div>'
    ),
}

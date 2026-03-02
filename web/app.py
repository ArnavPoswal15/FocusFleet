import gradio as gr
import os
import sqlite3
import datetime
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
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
    """
    if frame is None:
        return None
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
        return frame_disp

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

    return frame_disp

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

with gr.Blocks() as demo:
    gr.Markdown("## Driver Drowsiness Detection App")
    driver_state = gr.State(value=None)

    with gr.Tab("Login / Register"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Login")
                login_name = gr.Textbox(label="Name")
                login_password = gr.Textbox(label="Password", type="password")
                login_button = gr.Button("Login")
                login_status = gr.Textbox(label="Login Status", interactive=False)
            with gr.Column():
                gr.Markdown("### Register")
                reg_name = gr.Textbox(label="Name")
                reg_password = gr.Textbox(label="Password", type="password")
                reg_contact = gr.Textbox(label="Contact Info")
                reg_license = gr.Textbox(label="License Number")
                register_button = gr.Button("Register")
                reg_status = gr.Textbox(label="Registration Status", interactive=False)

        def login_callback(name, password):
            driver_info, msg = login_driver(name, password)
            return driver_info, msg

        login_button.click(login_callback, inputs=[login_name, login_password],
                           outputs=[driver_state, login_status])

        def register_callback(name, password, contact, license_num):
            return register_driver(name, password, contact, license_num)

        register_button.click(register_callback, inputs=[reg_name, reg_password, reg_contact, reg_license],
                              outputs=reg_status)

    with gr.Tab("Drowsiness Detection"):
        gr.Markdown("### Live Drowsiness Detection (Using EAR & Yawn)")
        welcome_msg = gr.Markdown(value="Please login to access this feature.")
        ear_threshold_slider = gr.Slider(minimum=0.1, maximum=0.5, value=0.25, label="EAR Threshold")
        # Larger webcam feed: 640x480
        webcam_input = gr.Image(sources=["webcam"], type="numpy", streaming=True,
                                label="Webcam Feed", height=480, width=640)
        detection_output = gr.Image(label="Detection Output", interactive=False,
                                    height=480, width=640)
        # Stream live frames (only if logged in)
        webcam_input.stream(fn=detect_drowsiness,
                            inputs=[webcam_input, ear_threshold_slider],
                            outputs=detection_output)

    with gr.Tab("Image Upload Classification"):
        gr.Markdown("### Upload an image to classify drowsiness")
        upload_input = gr.Image(type="numpy", label="Upload Image")
        upload_prediction = gr.Textbox(label="Prediction", interactive=False)
        upload_input.change(fn=classify_image, inputs=upload_input, outputs=upload_prediction)

    def update_welcome(driver_info):
        if driver_info is not None:
            return f"Welcome, {driver_info['Name']}! Monitor your drowsiness below."
        else:
            return "Please login to access drowsiness detection."

    driver_state.change(fn=update_welcome, inputs=driver_state, outputs=welcome_msg)

demo.launch()

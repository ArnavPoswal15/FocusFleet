import sys
import pathlib
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp

from threading import Lock

class ModelEngine:
    """Engine encapsulating ML model loading and inference logic.
    Provides methods for static image classification and live drowsiness detection.
    """

    def __init__(self, model_path: str = "driver_drowsiness_model.keras"):
        # Resolve model path robustly for both dev and PyInstaller bundled exe
        base_path = pathlib.Path(sys._MEIPASS) if getattr(sys, "_MEIPASS", False) else pathlib.Path(__file__).parent
        self.model_file = base_path / model_path
        self._load_model()
        
        # Load OpenCV Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # MediaPipe Face Mesh for live detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # Landmark indices
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        self.MOUTH_TOP_IDX = 13
        self.MOUTH_BOTTOM_IDX = 14
        self._lock = Lock()

    def _load_model(self):
        try:
            self.model = tf.keras.models.load_model(str(self.model_file))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_file}: {e}")

    @staticmethod
    def _euclidean_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _compute_ear(self, landmarks, eye_indices, w, h):
        pts = []
        for idx in eye_indices:
            lm = landmarks[idx]
            pts.append((int(lm.x * w), int(lm.y * h)))
        A = self._euclidean_distance(pts[1], pts[5])
        B = self._euclidean_distance(pts[2], pts[4])
        C = self._euclidean_distance(pts[0], pts[3])
        ear = (A + B) / (2.0 * C) if C != 0 else 0
        return ear

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize and expand dimensions for model prediction.
        Expects an RGB frame (numpy array).
        """
        try:
            resized = cv2.resize(frame, (224, 224))
            normalized = resized.astype("float32") / 255.0
            return np.expand_dims(normalized, axis=0)
        except Exception as e:
            raise RuntimeError(f"Image preprocessing failed: {e}")

    def predict_image(self, frame: np.ndarray) -> tuple[str, float]:
        """Classify a single image frame (RGB).
        Detects face using Haar Cascade, crops, resizes, and predicts.
        Returns a tuple of label ("Drowsy" or "Not Drowsy") and confidence percentage.
        Raises ValueError if no face is detected.
        """
        # Convert RGB to Grayscale for Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            raise ValueError("No face detected.")
            
        # Take the first face
        (x, y, w_box, h_box) = faces[0]
        face_roi = frame[y:y+h_box, x:x+w_box]

        with self._lock:
            input_tensor = self.preprocess_image(face_roi)
            preds = self.model.predict(input_tensor)
            idx = int(np.argmax(preds, axis=1)[0])
            confidence = float(preds[0][idx] * 100)
            label = "Drowsy" if idx == 0 else "Not Drowsy"
            return label, confidence

    def detect_drowsiness_live(self, frame: np.ndarray, ear_threshold: float = 0.25) -> tuple[np.ndarray, bool, dict]:
        """Process a live webcam frame, annotate it.
        Returns the annotated frame, a boolean (True if drowsy), 
        and a dictionary with metrics (avg_ear, mouth_dist) for logging.
        """
        # Convert BGR (OpenCV) to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        metrics = {"avg_ear": 0.0, "mouth_dist": 0.0, "face_detected": False}
        
        if not results.multi_face_landmarks:
            cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, False, metrics
        
        metrics["face_detected"] = True
            
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        left_ear = self._compute_ear(landmarks, self.LEFT_EYE_IDX, w, h)
        right_ear = self._compute_ear(landmarks, self.RIGHT_EYE_IDX, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Mouth distance
        lm_top = landmarks[self.MOUTH_TOP_IDX]
        lm_bottom = landmarks[self.MOUTH_BOTTOM_IDX]
        mouth_dist = self._euclidean_distance((lm_top.x * w, lm_top.y * h), (lm_bottom.x * w, lm_bottom.y * h))
        mouth_thresh = 0.08 * h
        
        drowsy = (avg_ear < ear_threshold) or (mouth_dist > mouth_thresh)
        metrics["avg_ear"] = avg_ear
        metrics["mouth_dist"] = mouth_dist
        
        # Bounding box
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x_min, x_max = int(min(xs) * w), int(max(xs) * w)
        y_min, y_max = int(min(ys) * h), int(max(ys) * h)
        color = (0, 0, 255) if drowsy else (0, 255, 0)
        label = f"{'Drowsy' if drowsy else 'Alert'}"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame, drowsy, metrics

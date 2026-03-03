import logging
import logging.handlers
import os
import time
from enum import Enum

class DriverState(Enum):
    ACTIVE = "ACTIVE"
    WARNING = "WARNING"
    DROWSY = "DROWSY"

class FaceState(Enum):
    DETECTED = "DETECTED"
    LOST = "LOST"

class YawnState(Enum):
    NOT_YAWNING = "NOT_YAWNING"
    YAWNING = "YAWNING"

class SessionLogger:
    def __init__(self, log_file_path: str):
        self.logger = logging.getLogger("DriverSession")
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent adding handlers multiple times
        if not self.logger.handlers:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            # Rotating file handler (5MB max, 3 backups)
            fh = logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=5*1024*1024, backupCount=3
            )
            # Default to INFO for production to avoid frame-level spam
            fh.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def info(self, msg: str):
        self.logger.info(msg)
        
    def warning(self, msg: str):
        self.logger.warning(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)
        
    def debug(self, msg: str):
        # Frame-level metrics (disabled in production since root is INFO)
        self.logger.debug(msg)


class DriverStateTracker:
    def __init__(self, log_file_path: str):
        self.logger = SessionLogger(log_file_path)
        
        # Core driver states
        self.current_state = DriverState.ACTIVE
        self.state_start_time = time.time()
        self.session_start_time = time.time()
        
        # Face detection states
        self.face_state = FaceState.DETECTED
        self.face_lost_start_time = None
        
        # Yawn states
        self.yawn_state = YawnState.NOT_YAWNING
        self.yawn_start_time = None
        self.max_mouth_ratio_current_yawn = 0.0
        
        # Analytics
        self.total_drowsy_events = 0
        self.total_warning_events = 0
        self.total_yawn_count = 0
        self.total_face_lost_events = 0
        
        self.total_drowsy_time = 0.0
        self.total_warning_time = 0.0
        self.total_active_time = 0.0
        
        # Summary interval tracking
        self.last_summary_time = time.time()
        self.summary_interval_seconds = 10 * 60 # 10 minutes
        
        # Settings
        self.warning_threshold = 1.0  # seconds EAR < thresh to WARNING
        self.drowsy_threshold = 2.0   # seconds EAR < thresh to DROWSY
        self.ear_threshold = 0.25 # Default
        
        # Track when EAR went below threshold for exact transition timing
        self.ear_below_start_time = None
        
        self.logger.info("SESSION_STARTED")

    def _update_drowsy_state(self, ear: float):
        now = time.time()
        
        # Check if EAR is below threshold
        is_eyes_closed = (ear < self.ear_threshold)
        
        if is_eyes_closed:
            if self.ear_below_start_time is None:
                self.ear_below_start_time = now
            
            elapsed_closed = now - self.ear_below_start_time
            
            if self.current_state == DriverState.ACTIVE and elapsed_closed >= self.warning_threshold:
                self._transition_to(DriverState.WARNING, ear)
            elif self.current_state == DriverState.WARNING and elapsed_closed >= self.drowsy_threshold:
                self._transition_to(DriverState.DROWSY, ear)
        else:
            # Eyes are open. Reset the timer.
            self.ear_below_start_time = None
            
            if self.current_state != DriverState.ACTIVE:
                self._transition_to(DriverState.ACTIVE, ear)
                
    def _transition_to(self, new_state: DriverState, ear: float):
        if self.current_state == new_state:
            return

        now = time.time()
        duration = now - self.state_start_time

        # Accumulate time for the outgoing state
        if self.current_state == DriverState.ACTIVE:
            self.total_active_time += duration
        elif self.current_state == DriverState.WARNING:
            self.total_warning_time += duration
        elif self.current_state == DriverState.DROWSY:
            self.total_drowsy_time += duration
            self.logger.info(f"Recovery Time: {duration:.1f} seconds") # Recovered from Drowsy

        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = now
        
        if new_state == DriverState.WARNING:
            self.total_warning_events += 1
            self.logger.info(f"STATE_CHANGE: {old_state.value} -> {new_state.value} | Reason: EAR below threshold for {self.warning_threshold}s | EAR: {ear:.3f}")
        elif new_state == DriverState.DROWSY:
            self.total_drowsy_events += 1
            self.logger.warning(f"STATE_CHANGE: {old_state.value} -> {new_state.value} | Reason: EAR below threshold for {self.drowsy_threshold}s")
        elif new_state == DriverState.ACTIVE:
            # If recovering from DROWSY
            if old_state == DriverState.DROWSY:
                self.logger.info(f"STATE_CHANGE: {old_state.value} -> {new_state.value} | Drowsy Duration: {duration:.1f} seconds")
            else:
                self.logger.info(f"STATE_CHANGE: {old_state.value} -> {new_state.value}")
                
    def _update_face_state(self, face_detected: bool):
        now = time.time()
        
        if not face_detected:
            if self.face_state == FaceState.DETECTED:
                self.face_state = FaceState.LOST
                self.face_lost_start_time = now
                self.total_face_lost_events += 1
                self.logger.warning("FACE_LOST")
                
                # If face is lost, we logically might want to reset the EAR timer
                self.ear_below_start_time = None
                
        else:
            if self.face_state == FaceState.LOST:
                self.face_state = FaceState.DETECTED
                duration = now - self.face_lost_start_time
                self.logger.info(f"FACE_RESTORED | Duration absence: {duration:.1f}s")
                self.face_lost_start_time = None
                
    def _update_yawn_state(self, is_yawning: bool, mouth_ratio: float):
        now = time.time()
        
        if is_yawning:
            if self.yawn_state == YawnState.NOT_YAWNING:
                self.yawn_state = YawnState.YAWNING
                self.yawn_start_time = now
                self.max_mouth_ratio_current_yawn = mouth_ratio
                self.total_yawn_count += 1
                self.logger.info("YAWN_STARTED")
            else:
                # Update max ratio if currently yawning
                if mouth_ratio > self.max_mouth_ratio_current_yawn:
                    self.max_mouth_ratio_current_yawn = mouth_ratio
        else:
            if self.yawn_state == YawnState.YAWNING:
                self.yawn_state = YawnState.NOT_YAWNING
                duration = now - self.yawn_start_time
                self.logger.info(f"YAWN_ENDED | Yawn Duration: {duration:.1f}s | Max Mouth Ratio: {self.max_mouth_ratio_current_yawn:.3f}")
                self.yawn_start_time = None
                self.max_mouth_ratio_current_yawn = 0.0

    def update_state(self, ear_threshold: float, face_detected: bool, metrics: dict):
        self.ear_threshold = ear_threshold
        
        ear = metrics.get('avg_ear', 0.0)
        mouth_dist = metrics.get('mouth_dist', 0.0)
        
        # Frame-level hidden logs
        self.logger.debug(f"Frame metrics - EAR: {ear:.3f}, Mouth: {mouth_dist:.1f}, Face: {face_detected}")

        now = time.time()
        
        # 1. Update Face State
        self._update_face_state(face_detected)
        
        # 2. Update Yawn and Drowsy states ONLY if face is detected
        if face_detected:
            # We assume a fixed face height approximation for mouth ratio since we just have mouth dist.
            # But the 'is_yawning' logic can be passed from the engine or app. 
            # We'll just take is_yawning as whether mouth>threshold. But wait, we don't have is_yawning passed in directly.
            # In the old logger, is_yawning was passed in. I changed the signature to remove is_drowsy and is_yawning
            # Wait, `app.py` computes `is_yawning = mouth_dist > (0.08 * h)`. Let's assume the caller passes that in.
            pass # We need to update the signature next line...

    def update_state_v2(self, ear_threshold: float, face_detected: bool, is_yawning: bool, metrics: dict):
        self.ear_threshold = ear_threshold
        ear = metrics.get('avg_ear', 0.0)
        mouth_dist = metrics.get('mouth_dist', 0.0)
        
        self.logger.debug(f"Frame metrics - EAR: {ear:.3f}, Mouth: {mouth_dist:.1f}, Face: {face_detected}")

        self._update_face_state(face_detected)
        
        if face_detected:
            self._update_drowsy_state(ear)
            # Use mouth_dist directly as the "ratio" proxy for max logging
            self._update_yawn_state(is_yawning, mouth_dist)
            
        # Check for summary
        self._check_summary()
        
    def _check_summary(self):
        now = time.time()
        if (now - self.last_summary_time) >= self.summary_interval_seconds:
            self._log_summary("PERIODIC SESSION SUMMARY")
            self.last_summary_time = now

    def _log_summary(self, title: str):
        # Make sure current state time is flushed to totals
        now = time.time()
        current_duration = now - self.state_start_time
        
        active = self.total_active_time + (current_duration if self.current_state == DriverState.ACTIVE else 0.0)
        warn = self.total_warning_time + (current_duration if self.current_state == DriverState.WARNING else 0.0)
        drowsy = self.total_drowsy_time + (current_duration if self.current_state == DriverState.DROWSY else 0.0)
        
        session_duration = now - self.session_start_time
        
        self.logger.info(f"--- {title} ---")
        self.logger.info(f"Session Duration: {session_duration:.1f}s")
        self.logger.info(f"Active Time: {active:.1f}s")
        self.logger.info(f"Warning Time: {warn:.1f}s")
        self.logger.info(f"Drowsy Time: {drowsy:.1f}s")
        self.logger.info(f"Total Drowsy Events: {self.total_drowsy_events}")
        self.logger.info(f"Total Warning Events: {self.total_warning_events}")
        self.logger.info(f"Total Yawn Events: {self.total_yawn_count}")
        self.logger.info(f"Face Lost Events: {self.total_face_lost_events}")
        self.logger.info(f"-------------------------")

    def end_session(self):
        self._log_summary("FINAL SESSION SUMMARY")
        self.logger.info("SESSION_ENDED")

    def log_error(self, msg: str):
        self.logger.error(msg)

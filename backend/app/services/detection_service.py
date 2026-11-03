"""Thin wrapper around focusfleet.detection for the FastAPI pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from focusfleet import config
from focusfleet.detection import process_frame
from focusfleet.state import STATE


@dataclass
class FrameResult:
    annotated_frame: np.ndarray | None
    driver_status: str
    ear: float | None
    yawn_detected: bool
    face_detected: bool
    inference_ms: float


class DetectionService:
    """Runs the same frame pipeline as the Gradio stream."""

    def process_frame(self, frame: np.ndarray | None, ear_threshold: float) -> FrameResult:
        started = time.perf_counter()

        if frame is None:
            return FrameResult(
                annotated_frame=None,
                driver_status=config.STATUS_NO_FACE,
                ear=None,
                yawn_detected=False,
                face_detected=False,
                inference_ms=0.0,
            )

        annotated = process_frame(frame, ear_threshold)
        metrics = STATE.last_frame_metrics or {}
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return FrameResult(
            annotated_frame=annotated,
            driver_status=STATE.current_status_key,
            ear=metrics.get("avg_ear"),
            yawn_detected=bool(metrics.get("is_yawning")),
            face_detected=bool(metrics.get("face_detected")),
            inference_ms=round(elapsed_ms, 2),
        )

    @staticmethod
    def encode_jpeg(frame: np.ndarray | None, quality: int = 85) -> bytes | None:
        if frame is None:
            return None
        ok, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        return buf.tobytes() if ok else None

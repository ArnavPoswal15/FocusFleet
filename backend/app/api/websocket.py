import asyncio
import base64
import datetime
import json

import cv2
import numpy as np
from fastapi import APIRouter, Cookie, Query, WebSocket, WebSocketDisconnect

from app.core.config import SESSION_COOKIE_NAME
from app.core.security import read_session_token
from app.services.detection_service import DetectionService
from focusfleet import config
from focusfleet.database import ROLE_DRIVER
from focusfleet.state import STATE

router = APIRouter(tags=["websocket"])

_detection = DetectionService()


def _decode_frame(payload: bytes) -> np.ndarray | None:
    arr = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@router.websocket("/ws/monitoring/{session_id}")
async def monitoring_ws(
    websocket: WebSocket,
    session_id: str,
    focusfleet_session: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    access_token: str | None = Query(default=None),
):
    token = focusfleet_session or access_token
    user = read_session_token(token) if token else None
    if not user:
        await websocket.close(code=4401)
        return
    if user.get("role", ROLE_DRIVER) != ROLE_DRIVER:
        await websocket.close(code=4403)
        return
    if STATE.session_id != session_id:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    frame_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=1)
    stop = asyncio.Event()

    async def reader() -> None:
        try:
            while not stop.is_set():
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"] is not None:
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    await frame_queue.put(message["bytes"])
        except WebSocketDisconnect:
            pass
        finally:
            stop.set()
            await frame_queue.put(None)

    async def processor() -> None:
        ear_threshold = config.DEFAULT_EAR_THRESHOLD
        if STATE.tracker is not None:
            ear_threshold = getattr(STATE.tracker, "ear_threshold", ear_threshold)

        while not stop.is_set():
            payload = await frame_queue.get()
            if payload is None:
                break

            frame = _decode_frame(payload)
            result = await asyncio.to_thread(_detection.process_frame, frame, ear_threshold)
            jpeg = await asyncio.to_thread(DetectionService.encode_jpeg, result.annotated_frame)

            body = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "driver_status": result.driver_status,
                "ear": result.ear,
                "yawn_detected": result.yawn_detected,
                "face_detected": result.face_detected,
                "inference_ms": result.inference_ms,
            }
            if jpeg:
                body["frame_jpeg_b64"] = base64.b64encode(jpeg).decode("ascii")

            await websocket.send_text(json.dumps(body))

    reader_task = asyncio.create_task(reader())
    processor_task = asyncio.create_task(processor())

    try:
        await asyncio.wait({reader_task, processor_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.set()
        reader_task.cancel()
        processor_task.cancel()
        if STATE.session_id == session_id and STATE.tracker is not None:
            STATE.tracker.end_session()
            STATE.tracker = None
            STATE.session_id = None
            STATE.log_file = None

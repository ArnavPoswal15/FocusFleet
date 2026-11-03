"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type CameraStatus = "idle" | "starting" | "active" | "denied" | "error";

const CAPTURE_INTERVAL_MS = 100;

export function useCamera(options?: {
  enabled?: boolean;
  onFrame?: (blob: Blob) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onFrameRef = useRef(options?.onFrame);
  onFrameRef.current = options?.onFrame;

  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const attachVideoRef = useCallback((node: HTMLVideoElement | null) => {
    videoRef.current = node;
    if (node && streamRef.current) {
      node.srcObject = streamRef.current;
    }
  }, []);

  const stopCaptureLoop = useCallback(() => {
    if (captureTimerRef.current != null) {
      clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }
  }, []);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    let canvas = canvasRef.current;
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvasRef.current = canvas;
    }

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;
    if (width === 0 || height === 0) return;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, width, height);
    canvas.toBlob(
      (blob) => {
        if (blob && onFrameRef.current) {
          onFrameRef.current(blob);
        }
      },
      "image/jpeg",
      0.72,
    );
  }, []);

  const startCaptureLoop = useCallback(() => {
    stopCaptureLoop();
    captureTimerRef.current = setInterval(captureFrame, CAPTURE_INTERVAL_MS);
  }, [captureFrame, stopCaptureLoop]);

  const start = useCallback(async () => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices) {
      setStatus("error");
      setError("Camera is not available in this environment.");
      return;
    }

    setStatus("starting");
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      setStatus("active");
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "Error";
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        setStatus("denied");
        setError("Camera permission was denied.");
      } else {
        setStatus("error");
        setError("Unable to start the camera.");
      }
    }
  }, []);

  const stop = useCallback(() => {
    stopCaptureLoop();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus("idle");
  }, [stopCaptureLoop]);

  useEffect(() => {
    if (options?.enabled && status === "active") {
      startCaptureLoop();
      return stopCaptureLoop;
    }
    stopCaptureLoop();
    return undefined;
  }, [options?.enabled, status, startCaptureLoop, stopCaptureLoop]);

  useEffect(() => {
    return () => {
      stopCaptureLoop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [stopCaptureLoop]);

  return {
    attachVideoRef,
    start,
    stop,
    status,
    error,
    isActive: status === "active",
  };
}

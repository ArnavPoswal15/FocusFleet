"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, USE_MOCK, api } from "@/lib/api";
import {
  frameFromBackend,
  mockMonitoringClient,
  type BackendMonitoringPayload,
  type MonitoringFrame,
  type DriverState,
} from "@/lib/mock-monitoring";
import { WebSocketClient } from "@/lib/websocket";

function wsBase(): string {
  const configured = process.env.NEXT_PUBLIC_WS_URL?.replace(/\/$/, "");
  if (configured) {
    return configured;
  }
  if (API_BASE) {
    return API_BASE.replace(/^http/, "ws");
  }
  if (typeof window !== "undefined") {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}`;
  }
  return "ws://localhost:8000";
}

export type MonitoringConnection = "idle" | "connecting" | "connected" | "disconnected";

export function useMonitoring(sessionId: string | null) {
  const [connection, setConnection] = useState<MonitoringConnection>("idle");
  const [frame, setFrame] = useState<MonitoringFrame | null>(null);
  const clientRef = useRef<WebSocketClient | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setConnection("idle");
      setFrame(null);
      return;
    }

    if (USE_MOCK) {
      const onFrame = (next: MonitoringFrame) => setFrame(next);
      mockMonitoringClient.connect();
      setConnection("connected");
      const unsubscribe = mockMonitoringClient.subscribe(onFrame);
      return () => {
        unsubscribe();
        mockMonitoringClient.disconnect();
        setConnection("idle");
        setFrame(null);
      };
    }

    const url = `${wsBase()}/ws/monitoring/${encodeURIComponent(sessionId)}`;
    const client = new WebSocketClient({ url, reconnectMs: 2500 });
    clientRef.current = client;
    setConnection("connecting");

    const offOpen = client.on("open", () => setConnection("connected"));
    const offClose = client.on("close", () => setConnection("disconnected"));
    const offFrame = client.on<BackendMonitoringPayload>("frame", (payload) => {
      setFrame(frameFromBackend(payload));
    });

    client.connect();

    return () => {
      offOpen();
      offClose();
      offFrame();
      client.disconnect();
      clientRef.current = null;
      setConnection("idle");
      setFrame(null);
    };
  }, [sessionId]);

  const sendFrame = useCallback((data: ArrayBuffer | Blob) => {
    clientRef.current?.sendBytes(data);
  }, []);

  const state: DriverState = frame?.state ?? "unknown";
  const connected = connection === "connected";

  return {
    connected,
    connection,
    frame,
    state,
    ear: frame?.ear ?? null,
    confidence: frame?.confidence ?? null,
    inferenceMs: frame?.inference_ms ?? null,
    frameJpegB64: frame?.frame_jpeg_b64 ?? null,
    faceDetected: frame?.face_detected ?? null,
    driverStatus: frame?.driver_status ?? null,
    sendFrame,
    stopSession: async (id: string) => api.stopSession(id),
  };
}

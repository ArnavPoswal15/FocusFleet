/**
 * Mock WebSocket-style monitoring client for development without a backend.
 */

export type DriverState = "focused" | "distracted" | "drowsy" | "unknown";

export type MonitoringFrame = {
  timestamp: number | string;
  state: DriverState;
  ear: number | null;
  confidence: number | null;
  inference_ms?: number;
  frame_jpeg_b64?: string;
  driver_status?: string;
  face_detected?: boolean;
  yawn_detected?: boolean;
};

export type BackendMonitoringPayload = {
  timestamp?: string;
  driver_status?: string;
  ear?: number | null;
  yawn_detected?: boolean;
  face_detected?: boolean;
  inference_ms?: number;
  frame_jpeg_b64?: string;
};

export function mapDriverStatus(status?: string): DriverState {
  switch (status) {
    case "drowsy":
      return "drowsy";
    case "warning":
      return "distracted";
    case "alert":
    case "neutral":
      return "focused";
    case "no_face":
    default:
      return "unknown";
  }
}

export function frameFromBackend(payload: BackendMonitoringPayload): MonitoringFrame {
  const state = mapDriverStatus(payload.driver_status);
  return {
    timestamp: payload.timestamp ?? Date.now(),
    state,
    ear: payload.ear ?? null,
    confidence:
      payload.face_detected == null ? null : payload.face_detected ? 0.92 : 0.35,
    inference_ms: payload.inference_ms,
    frame_jpeg_b64: payload.frame_jpeg_b64,
    driver_status: payload.driver_status,
    face_detected: payload.face_detected,
    yawn_detected: payload.yawn_detected,
  };
}

export type MonitoringListener = (frame: MonitoringFrame) => void;

const STATES: DriverState[] = ["focused", "focused", "focused", "distracted", "drowsy"];

function randomEar(state: DriverState): number {
  switch (state) {
    case "drowsy":
      return 0.12 + Math.random() * 0.05;
    case "distracted":
      return 0.18 + Math.random() * 0.06;
    case "focused":
      return 0.28 + Math.random() * 0.08;
    default:
      return 0.2;
  }
}

export class MockMonitoringClient {
  private listeners = new Set<MonitoringListener>();
  private timer: ReturnType<typeof setInterval> | null = null;
  private connected = false;

  connect(): void {
    if (this.connected) return;
    this.connected = true;
    this.timer = setInterval(() => {
      const state = STATES[Math.floor(Math.random() * STATES.length)];
      const frame: MonitoringFrame = {
        timestamp: Date.now(),
        state,
        ear: randomEar(state),
        confidence: 0.75 + Math.random() * 0.2,
      };
      this.listeners.forEach((fn) => fn(frame));
    }, 1200);
  }

  disconnect(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.connected = false;
  }

  subscribe(listener: MonitoringListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  get isConnected(): boolean {
    return this.connected;
  }
}

export const mockMonitoringClient = new MockMonitoringClient();

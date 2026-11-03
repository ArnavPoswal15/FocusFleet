"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DetectionStatusBanner,
  mapStatusToBanner,
} from "@/components/monitoring/detection-status-banner";
import { PageError } from "@/components/ui/page-state";
import { useCamera } from "@/hooks/use-camera";
import { useMonitoring, type MonitoringConnection } from "@/hooks/use-monitoring";
import { api, USE_MOCK, type ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";

const STATE_LABEL: Record<string, string> = {
  focused: "Focused",
  distracted: "Distracted",
  drowsy: "Drowsy",
  unknown: "Waiting",
};

const CONNECTION_LABEL: Record<MonitoringConnection, string> = {
  idle: "Not connected",
  connecting: "Connecting…",
  connected: "Connected",
  disconnected: "Disconnected",
};

export function MonitoringPanel() {
  const [sessionActive, setSessionActive] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const monitoring = useMonitoring(sessionId);

  const sendFrame = useCallback(
    (blob: Blob) => {
      if (!USE_MOCK && sessionId) {
        monitoring.sendFrame(blob);
      }
    },
    [monitoring, sessionId],
  );

  const camera = useCamera({
    enabled: sessionActive && cameraShouldStream(sessionId, monitoring.connection),
    onFrame: sendFrame,
  });

  useEffect(() => {
    if (sessionActive && !camera.isActive && camera.status === "idle") {
      void camera.start();
    }
  }, [sessionActive, camera]);

  const previewSrc = useMemo(() => {
    if (monitoring.frameJpegB64) {
      return `data:image/jpeg;base64,${monitoring.frameJpegB64}`;
    }
    return null;
  }, [monitoring.frameJpegB64]);

  const bannerVariant = mapStatusToBanner(
    monitoring.driverStatus ?? undefined,
    monitoring.state,
  );

  const showAnnotated = Boolean(previewSrc) && !USE_MOCK;

  const toggleSession = async () => {
    if (sessionActive) {
      setBusy(true);
      try {
        if (sessionId && !USE_MOCK) {
          await monitoring.stopSession(sessionId);
        }
      } catch {
        // session may already be closed server-side
      } finally {
        camera.stop();
        setSessionActive(false);
        setSessionId(null);
        setBusy(false);
      }
      return;
    }

    setBusy(true);
    setSessionError(null);
    try {
      const { session_id } = await api.startSession();
      setSessionId(session_id);
      setSessionActive(true);
    } catch (err) {
      const apiErr = err as ApiError;
      setSessionError(apiErr.message ?? "Could not start session");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
      <Card className="overflow-hidden border-border bg-card">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
          <CardTitle className="text-[13px] font-medium text-muted-foreground">
            Live preview
          </CardTitle>
          <Badge variant="outline" className="border-border font-normal">
            {camera.status === "active" ? "Camera on" : "Camera off"}
          </Badge>
        </CardHeader>
        <CardContent className="space-y-4">
          <DetectionStatusBanner variant={sessionActive ? bannerVariant : "neutral"} />
          {USE_MOCK ? (
            <p className="text-[12px] text-muted-foreground">
              Mock mode shows metrics only. Set{" "}
              <code className="rounded bg-white/5 px-1">NEXT_PUBLIC_USE_MOCK=false</code> and
              run the Python backend for the face box overlay.
            </p>
          ) : null}
          <div className="relative aspect-video overflow-hidden rounded-xl border border-border bg-black">
            <video
              ref={camera.attachVideoRef}
              className={cn(
                "size-full object-cover",
                showAnnotated ? "opacity-0" : "opacity-100",
              )}
              playsInline
              muted
              autoPlay
            />
            {showAnnotated ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={previewSrc!}
                alt="Face detection with bounding box and drowsiness label"
                className="absolute inset-0 size-full object-cover"
              />
            ) : null}
            {sessionActive && !USE_MOCK && camera.isActive && !showAnnotated ? (
              <div className="absolute inset-x-0 bottom-0 bg-black/60 px-3 py-2 text-[12px] text-muted-foreground">
                Processing first frame…
              </div>
            ) : null}
            {!camera.isActive ? (
              <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-[13px] text-muted-foreground">
                {USE_MOCK
                  ? "Start a session to enable the camera (mock metrics only)"
                  : "Log in, then start a session to run face detection"}
              </div>
            ) : null}
          </div>
          {camera.error ? (
            <p className="text-[13px] text-destructive">{camera.error}</p>
          ) : null}
          {sessionError ? <PageError message={sessionError} /> : null}
          <div className="flex gap-2">
            <Button
              type="button"
              disabled={busy}
              onClick={() => void toggleSession()}
              className={cn(
                sessionActive
                  ? "bg-secondary text-secondary-foreground hover:bg-secondary/90"
                  : "bg-accent text-accent-foreground hover:bg-accent/90",
              )}
            >
              {busy ? "Please wait…" : sessionActive ? "Stop session" : "Start session"}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="border-border bg-card">
        <CardHeader className="pb-3">
          <CardTitle className="text-[13px] font-medium text-muted-foreground">
            Detection status
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <div>
            <p className="text-[13px] text-muted-foreground">Driver state</p>
            <p className="mt-1 text-2xl font-semibold tracking-tight">
              {STATE_LABEL[monitoring.state] ?? monitoring.state}
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-[13px] text-muted-foreground">EAR</p>
              <p className="mt-1 text-lg font-medium tabular-nums">
                {monitoring.ear != null ? monitoring.ear.toFixed(3) : "—"}
              </p>
            </div>
            <div>
              <p className="text-[13px] text-muted-foreground">Face</p>
              <p className="mt-1 text-lg font-medium tabular-nums">
                {monitoring.faceDetected == null
                  ? "—"
                  : monitoring.faceDetected
                    ? "Detected"
                    : "Not detected"}
              </p>
            </div>
            <div>
              <p className="text-[13px] text-muted-foreground">Inference</p>
              <p className="mt-1 text-lg font-medium tabular-nums">
                {monitoring.inferenceMs != null
                  ? `${monitoring.inferenceMs.toFixed(1)} ms`
                  : "—"}
              </p>
            </div>
          </div>
          <div className="flex items-center justify-between rounded-xl border border-border px-3 py-2.5">
            <span className="text-[13px] text-muted-foreground">Stream</span>
            <span className="text-[13px] text-foreground">
              {USE_MOCK ? "Mock feed" : CONNECTION_LABEL[monitoring.connection]}
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function cameraShouldStream(
  sessionId: string | null,
  connection: MonitoringConnection,
): boolean {
  if (!sessionId) return false;
  if (USE_MOCK) return true;
  return connection === "connected" || connection === "connecting";
}

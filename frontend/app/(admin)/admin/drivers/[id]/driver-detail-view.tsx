"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageError, PageLoading } from "@/components/ui/page-state";
import { api, type ApiError } from "@/lib/api";
import type { AdminDriverDetail, SessionInfo } from "@/lib/analytics-types";

export function DriverDetailView() {
  const params = useParams();
  const idParam = params.id;
  const driverId =
    typeof idParam === "string" ? Number(idParam) : Number(idParam?.[0]);

  const [driver, setDriver] = useState<AdminDriverDetail | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isFinite(driverId)) {
      setError("Invalid driver id");
      setLoading(false);
      return;
    }

    void (async () => {
      setLoading(true);
      try {
        const [profile, sessionList] = await Promise.all([
          api.admin.getDriver(driverId),
          api.admin.listSessions(driverId),
        ]);
        setDriver(profile);
        setSessions(sessionList.sessions);
      } catch (err) {
        const apiErr = err as ApiError;
        setError(apiErr.message ?? "Failed to load driver");
      } finally {
        setLoading(false);
      }
    })();
  }, [driverId]);

  if (loading) return <PageLoading />;
  if (error || !driver) return <PageError message={error ?? "Driver not found"} />;

  return (
    <div className="space-y-6">
      <div>
        <Link
          href="/admin/drivers/"
          className="text-[13px] text-muted-foreground underline-offset-4 hover:underline"
        >
          ← All drivers
        </Link>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight">{driver.name}</h1>
        <p className="mt-1 text-[13px] text-muted-foreground">
          {driver.contact_info || "No contact"} · License {driver.license_number || "—"}
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "Sessions", value: driver.session_count },
          { label: "Drowsy events", value: driver.total_drowsy_events },
          { label: "Yawns", value: driver.total_yawns },
          {
            label: "Last session",
            value:
              driver.last_session_duration_s != null
                ? `${Math.round(driver.last_session_duration_s / 60)} min`
                : "—",
          },
        ].map((stat) => (
          <Card key={stat.label} className="border-border bg-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-[13px] font-medium text-muted-foreground">
                {stat.label}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-xl font-semibold tabular-nums">{stat.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-base font-semibold">Session logs</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {sessions.length === 0 ? (
            <p className="text-[13px] text-muted-foreground">No sessions recorded yet.</p>
          ) : (
            sessions.map((session) => (
              <Link
                key={session.session_id}
                href={`/admin/drivers/${driver.driver_id}/sessions/${session.session_id}/`}
                className="flex items-center justify-between rounded-lg border border-border/60 px-3 py-2.5 text-[13px] transition-colors hover:bg-white/[0.04]"
              >
                <span className="font-mono text-foreground">{session.session_id}</span>
                <span className="text-muted-foreground">
                  {session.active ? "Active" : session.log_file ? "View log" : "No log file"}
                </span>
              </Link>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}

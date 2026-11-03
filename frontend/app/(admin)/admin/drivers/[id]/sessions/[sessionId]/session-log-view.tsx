"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageError, PageLoading } from "@/components/ui/page-state";
import { api, type ApiError } from "@/lib/api";

export function SessionLogView() {
  const params = useParams();
  const idParam = params.id;
  const sessionParam = params.sessionId;
  const driverId =
    typeof idParam === "string" ? Number(idParam) : Number(idParam?.[0]);
  const sessionId =
    typeof sessionParam === "string" ? sessionParam : sessionParam?.[0] ?? "";

  const [log, setLog] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isFinite(driverId) || !sessionId) {
      setError("Invalid session");
      setLoading(false);
      return;
    }

    void (async () => {
      setLoading(true);
      try {
        const data = await api.admin.getSessionLog(driverId, sessionId, 500);
        setLog(data.content);
      } catch (err) {
        const apiErr = err as ApiError;
        setError(apiErr.message ?? "Failed to load log");
      } finally {
        setLoading(false);
      }
    })();
  }, [driverId, sessionId]);

  if (loading) return <PageLoading />;
  if (error) return <PageError message={error} />;

  return (
    <div className="space-y-6">
      <div>
        <Link
          href={`/admin/drivers/${driverId}/`}
          className="text-[13px] text-muted-foreground underline-offset-4 hover:underline"
        >
          ← Driver sessions
        </Link>
        <h1 className="mt-2 font-mono text-xl font-semibold tracking-tight">{sessionId}</h1>
      </div>

      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-base font-semibold">Session log</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="max-h-[70vh] overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border/60 bg-black/30 p-4 font-mono text-[12px] leading-relaxed text-foreground">
            {log || "(empty log)"}
          </pre>
        </CardContent>
      </Card>
    </div>
  );
}

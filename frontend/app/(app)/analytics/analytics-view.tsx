"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageEmpty, PageError, PageLoading } from "@/components/ui/page-state";
import { api, type ApiError } from "@/lib/api";
import type { AnalyticsData } from "@/lib/analytics-types";

export function AnalyticsView() {
  const router = useRouter();
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.getAnalytics());
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr.status === 401) {
        router.replace("/login/");
        return;
      }
      setError(apiErr.message ?? "Unable to load analytics");
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) return <PageLoading />;
  if (error) return <PageError message={error} onRetry={() => void load()} />;
  if (!data) return null;

  const { weekly, summary } = data;
  const hasWeekly = weekly.length > 0;

  return (
    <>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <SummaryCard label="Sessions" value={String(summary.session_count)} />
        <SummaryCard label="Drowsy events" value={String(summary.total_drowsy_events)} />
        <SummaryCard label="Yawns" value={String(summary.total_yawns)} />
        <SummaryCard
          label="Last session"
          value={
            summary.last_session_duration_s != null
              ? `${Math.round(summary.last_session_duration_s / 60)} min`
              : "—"
          }
        />
      </div>

      <Card className="mt-6 border-border bg-card">
        <CardHeader>
          <CardTitle className="text-[13px] font-medium text-muted-foreground">
            Sessions and alerts by day
          </CardTitle>
        </CardHeader>
        <CardContent className="h-64">
          {hasWeekly ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={weekly} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                <XAxis
                  dataKey="day"
                  tick={{ fill: "rgba(255,255,255,0.45)", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  allowDecimals={false}
                  tick={{ fill: "rgba(255,255,255,0.45)", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  contentStyle={{
                    background: "#1c1c1e",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: 12,
                    fontSize: 12,
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar
                  dataKey="sessions"
                  name="Sessions"
                  fill="rgba(48, 213, 200, 0.85)"
                  radius={[6, 6, 0, 0]}
                />
                <Bar
                  dataKey="alerts"
                  name="Alerts"
                  fill="rgba(255, 159, 10, 0.75)"
                  radius={[6, 6, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <PageEmpty
              title="No daily breakdown yet"
              description="Weekly charts appear when daily aggregates are available. Summary totals above reflect your saved sessions."
            />
          )}
        </CardContent>
      </Card>
    </>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <Card className="border-border bg-card">
      <CardHeader className="pb-2">
        <CardTitle className="text-[13px] font-medium text-muted-foreground">
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-2xl font-semibold tracking-tight">{value}</p>
      </CardContent>
    </Card>
  );
}

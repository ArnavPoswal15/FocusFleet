"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageEmpty, PageError, PageLoading } from "@/components/ui/page-state";
import { api, type ApiError } from "@/lib/api";
import type { ApiUser, DashboardStat, RecentEvent } from "@/lib/analytics-types";

export function DashboardView() {
  const router = useRouter();
  const [user, setUser] = useState<ApiUser | null>(null);
  const [stats, setStats] = useState<DashboardStat[]>([]);
  const [events, setEvents] = useState<RecentEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [session, dashboardStats, recent] = await Promise.all([
        api.getSession(),
        api.getDashboardStats(),
        api.getRecentEvents(),
      ]);
      setUser(session.user);
      setStats(dashboardStats);
      setEvents(recent);
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr.status === 401) {
        router.replace("/login/");
        return;
      }
      setError(apiErr.message ?? "Unable to load dashboard");
    } finally {
      setLoading(false);
    }
  }, [router]);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) return <PageLoading />;
  if (error) return <PageError message={error} onRetry={() => void load()} />;

  const displayName = user?.name ?? "Driver";

  return (
    <>
      <p className="-mt-6 mb-8 text-[13px] text-muted-foreground">
        Welcome back, {displayName}.
      </p>

      <div className="grid gap-4 sm:grid-cols-2">
        {stats.map((stat) => (
          <Card
            key={stat.id}
            className="border-border bg-card transition-colors duration-150"
          >
            <CardHeader className="pb-2">
              <CardTitle className="text-[13px] font-medium text-muted-foreground">
                {stat.label}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-1">
              <p className="text-3xl font-semibold tracking-tight">{stat.value}</p>
              <p className="text-[13px] text-muted-foreground">{stat.detail}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <section className="mt-10">
        <h2 className="mb-4 text-[13px] font-medium uppercase tracking-wide text-muted-foreground">
          Recent activity
        </h2>
        {events.length === 0 ? (
          <PageEmpty
            title="No sessions yet"
            description="Start a monitoring session to build your activity history."
          />
        ) : (
          <div className="divide-y divide-border rounded-2xl border border-border bg-card">
            {events.map((event) => (
              <div
                key={event.id}
                className="flex items-center justify-between gap-4 px-4 py-3.5 transition-colors duration-150 hover:bg-white/[0.02]"
              >
                <div>
                  <p className="text-sm text-foreground">{event.message}</p>
                  <p className="text-[13px] text-muted-foreground">{event.time}</p>
                </div>
                <Badge
                  variant="outline"
                  className="border-border text-[11px] font-normal capitalize"
                >
                  {event.type}
                </Badge>
              </div>
            ))}
          </div>
        )}
      </section>
    </>
  );
}

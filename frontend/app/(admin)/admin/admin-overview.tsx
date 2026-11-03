"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api, type ApiError } from "@/lib/api";
import type { FleetSummary } from "@/lib/analytics-types";

export function AdminOverview() {
  const [summary, setSummary] = useState<FleetSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const data = await api.admin.fleetSummary();
        setSummary(data);
      } catch (err) {
        const apiErr = err as ApiError;
        setError(apiErr.message ?? "Failed to load fleet summary");
      }
    })();
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Fleet overview</h1>
        <p className="mt-1 text-[13px] text-muted-foreground">
          Monitor drivers, sessions, and safety events across the fleet.
        </p>
      </div>

      {error ? <p className="text-[13px] text-destructive">{error}</p> : null}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: "Drivers", value: summary?.driver_count ?? "—" },
          { label: "Sessions", value: summary?.session_count ?? "—" },
          { label: "Drowsy events", value: summary?.total_drowsy_events ?? "—" },
          { label: "Yawn events", value: summary?.total_yawns ?? "—" },
        ].map((stat) => (
          <Card key={stat.label} className="border-border bg-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-[13px] font-medium text-muted-foreground">
                {stat.label}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-semibold tabular-nums">{stat.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-base font-semibold">Manage drivers</CardTitle>
        </CardHeader>
        <CardContent>
          <Link
            href="/admin/drivers/"
            className="text-[13px] text-foreground underline-offset-4 hover:underline"
          >
            View all drivers and session logs →
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}

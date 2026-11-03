"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageError, PageLoading } from "@/components/ui/page-state";
import { api, type ApiError } from "@/lib/api";
import type { AdminDriver } from "@/lib/analytics-types";

export function DriversList() {
  const [drivers, setDrivers] = useState<AdminDriver[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      setLoading(true);
      try {
        const { drivers: rows } = await api.admin.listDrivers();
        setDrivers(rows);
      } catch (err) {
        const apiErr = err as ApiError;
        setError(apiErr.message ?? "Failed to load drivers");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return drivers;
    return drivers.filter(
      (d) =>
        d.name.toLowerCase().includes(q) ||
        (d.license_number ?? "").toLowerCase().includes(q) ||
        (d.contact_info ?? "").toLowerCase().includes(q),
    );
  }, [drivers, query]);

  if (loading) return <PageLoading />;
  if (error) return <PageError message={error} />;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Drivers</h1>
        <p className="mt-1 text-[13px] text-muted-foreground">
          Registered fleet drivers and session counts.
        </p>
      </div>

      <Input
        placeholder="Search by name, license, or contact…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="max-w-md border-border bg-background"
      />

      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-base font-semibold">
            {filtered.length} driver{filtered.length === 1 ? "" : "s"}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {filtered.length === 0 ? (
            <p className="text-[13px] text-muted-foreground">No drivers match your search.</p>
          ) : (
            filtered.map((driver) => (
              <Link
                key={driver.driver_id}
                href={`/admin/drivers/${driver.driver_id}/`}
                className="flex items-center justify-between rounded-lg border border-border/60 px-3 py-2.5 text-[13px] transition-colors hover:bg-white/[0.04]"
              >
                <span className="font-medium text-foreground">{driver.name}</span>
                <span className="text-muted-foreground">
                  {driver.session_count} session{driver.session_count === 1 ? "" : "s"}
                </span>
              </Link>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}

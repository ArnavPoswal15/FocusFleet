"use client";

import { useCallback, useEffect, useState } from "react";
import { PageError, PageLoading } from "@/components/ui/page-state";
import { SettingsForm } from "./settings-form";
import { api, type ApiError } from "@/lib/api";
import type { MockSettings } from "@/lib/mock-data";

export function SettingsView() {
  const [settings, setSettings] = useState<MockSettings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSettings(await api.getSettings());
    } catch (err) {
      const apiErr = err as ApiError;
      setError(apiErr.message ?? "Unable to load settings");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (loading) return <PageLoading className="max-w-xl" />;
  if (error) return <PageError message={error} onRetry={() => void load()} />;
  if (!settings) return null;

  return <SettingsForm initial={settings} />;
}

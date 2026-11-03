"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import type { MockSettings } from "@/lib/mock-data";

export function SettingsForm({ initial }: { initial: MockSettings }) {
  const [settings, setSettings] = useState(initial);
  const [saved, setSaved] = useState(false);

  const save = () => {
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2000);
  };

  return (
    <Card className="max-w-xl border-border bg-card">
      <CardHeader>
        <CardTitle className="text-base font-semibold">Preferences</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex items-center justify-between gap-4">
          <div>
            <Label htmlFor="alert-sound" className="text-sm">
              Alert sound
            </Label>
            <p className="text-[13px] text-muted-foreground">
              Play audio when drowsiness is detected
            </p>
          </div>
          <input
            id="alert-sound"
            type="checkbox"
            checked={settings.alertSound}
            onChange={(e) =>
              setSettings((s) => ({ ...s, alertSound: e.target.checked }))
            }
            className="size-4 rounded border-border accent-[#30d5c8]"
          />
        </div>

        <Separator className="bg-border" />

        <div className="space-y-2">
          <Label htmlFor="sensitivity" className="text-sm">
            Detection sensitivity
          </Label>
          <select
            id="sensitivity"
            value={settings.sensitivity}
            onChange={(e) =>
              setSettings((s) => ({
                ...s,
                sensitivity: e.target.value as MockSettings["sensitivity"],
              }))
            }
            className="h-9 w-full rounded-xl border border-border bg-background px-3 text-sm transition-colors duration-150 outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="retention" className="text-sm">
            Data retention (days)
          </Label>
          <Input
            id="retention"
            type="number"
            min={7}
            max={365}
            value={settings.dataRetentionDays}
            onChange={(e) =>
              setSettings((s) => ({
                ...s,
                dataRetentionDays: Number(e.target.value) || s.dataRetentionDays,
              }))
            }
            className="border-border bg-background"
          />
        </div>

        <div className="flex items-center gap-3 pt-2">
          <Button type="button" variant="secondary" onClick={save}>
            Save changes
          </Button>
          {saved ? (
            <span className="text-[13px] text-muted-foreground">Saved</span>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

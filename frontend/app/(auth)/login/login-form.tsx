"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, type ApiError } from "@/lib/api";

export function LoginForm() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const { user } = await api.login(name, password);
      router.push(user.role === "admin" ? "/admin/" : "/dashboard/");
    } catch (err) {
      const apiErr = err as ApiError;
      setError(apiErr.message ?? "Sign in failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="border-border bg-card">
      <CardHeader className="space-y-1">
        <p className="text-[13px] font-medium text-muted-foreground">FocusFleet</p>
        <CardTitle className="text-2xl font-semibold tracking-tight">Sign in</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Username</Label>
            <Input
              id="name"
              autoComplete="username"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="border-border bg-background"
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="border-border bg-background"
              required
            />
          </div>
          {error ? <p className="text-[13px] text-destructive">{error}</p> : null}
          <Button
            type="submit"
            disabled={loading}
            className="w-full bg-accent text-accent-foreground hover:bg-accent/90"
          >
            {loading ? "Signing in…" : "Continue"}
          </Button>
        </form>
        <p className="mt-6 text-center text-[13px] text-muted-foreground">
          No account?{" "}
          <Link href="/register/" className="text-foreground underline-offset-4 hover:underline">
            Create one
          </Link>
        </p>
      </CardContent>
    </Card>
  );
}

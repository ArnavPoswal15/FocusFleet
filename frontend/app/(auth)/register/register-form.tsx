"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, type ApiError } from "@/lib/api";

export function RegisterForm() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [contact, setContact] = useState("");
  const [licenseNumber, setLicenseNumber] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await api.register(name, password, contact, licenseNumber);
      router.push("/dashboard/");
    } catch (err) {
      const apiErr = err as ApiError;
      setError(apiErr.message ?? "Registration failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="border-border bg-card">
      <CardHeader className="space-y-1">
        <p className="text-[13px] font-medium text-muted-foreground">FocusFleet</p>
        <CardTitle className="text-2xl font-semibold tracking-tight">Create account</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Full name</Label>
            <Input
              id="name"
              autoComplete="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="border-border bg-background"
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="contact">Contact (optional)</Label>
            <Input
              id="contact"
              autoComplete="email"
              value={contact}
              onChange={(e) => setContact(e.target.value)}
              className="border-border bg-background"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="license">License number (optional)</Label>
            <Input
              id="license"
              value={licenseNumber}
              onChange={(e) => setLicenseNumber(e.target.value)}
              className="border-border bg-background"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              autoComplete="new-password"
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
            {loading ? "Creating…" : "Register"}
          </Button>
        </form>
        <p className="mt-6 text-center text-[13px] text-muted-foreground">
          Already registered?{" "}
          <Link href="/login/" className="text-foreground underline-offset-4 hover:underline">
            Sign in
          </Link>
        </p>
      </CardContent>
    </Card>
  );
}

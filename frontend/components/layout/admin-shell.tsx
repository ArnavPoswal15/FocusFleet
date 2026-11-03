"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { LayoutDashboard, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, USE_MOCK, type ApiError } from "@/lib/api";
import type { ApiUser } from "@/lib/analytics-types";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/admin/", label: "Overview", icon: LayoutDashboard },
  { href: "/admin/drivers/", label: "Drivers", icon: Users },
] as const;

export function AdminShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<ApiUser | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);

  const refreshSession = useCallback(async () => {
    try {
      const session = await api.getSession();
      if (session.user.role !== "admin") {
        router.replace("/login/");
        return;
      }
      setUser(session.user);
    } catch {
      router.replace("/login/");
    } finally {
      setAuthReady(true);
    }
  }, [router]);

  useEffect(() => {
    setAuthReady(false);
    void refreshSession();
  }, [pathname, refreshSession]);

  const handleLogout = async () => {
    setLoggingOut(true);
    try {
      await api.logout();
      setUser(null);
      router.push("/login/");
    } catch (err) {
      const apiErr = err as ApiError;
      console.error(apiErr.message ?? "Logout failed");
    } finally {
      setLoggingOut(false);
    }
  };

  if (!authReady) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background text-muted-foreground">
        Loading…
      </div>
    );
  }

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <aside className="flex w-[220px] shrink-0 flex-col border-r border-border bg-[#1c1c1e]/40 px-3 py-5">
        <div className="mb-8 px-2">
          <p className="text-[13px] font-medium tracking-wide text-muted-foreground">
            FocusFleet
          </p>
          <p className="text-sm font-semibold text-foreground">Fleet admin</p>
        </div>

        <nav className="flex flex-1 flex-col gap-0.5">
          {NAV.map(({ href, label, icon: Icon }) => {
            const path = pathname.endsWith("/") ? pathname : `${pathname}/`;
            const active =
              href === "/admin/"
                ? path === "/admin/" || path === "/admin"
                : path.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "flex items-center gap-2.5 rounded-xl px-2.5 py-2 text-[13px] font-medium transition-[background-color,color] duration-150",
                  active
                    ? "bg-white/[0.08] text-foreground shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]"
                    : "text-muted-foreground hover:bg-white/[0.04] hover:text-foreground",
                )}
              >
                <Icon
                  className={cn(
                    "size-[18px] stroke-[1.5]",
                    active ? "text-accent" : "text-muted-foreground",
                  )}
                  aria-hidden
                />
                {label}
              </Link>
            );
          })}
        </nav>

        <p className="px-2 pt-4 text-[11px] text-muted-foreground/80">
          {USE_MOCK ? "Development data" : "Live API"}
        </p>
      </aside>

      <div className="flex min-h-screen flex-1 flex-col overflow-hidden">
        <header className="flex shrink-0 items-center justify-end gap-3 border-b border-border bg-background/80 px-10 py-3 backdrop-blur-sm">
          {user ? (
            <>
              <span className="max-w-[200px] truncate text-[13px] text-muted-foreground">
                {user.name}
              </span>
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={loggingOut}
                onClick={() => void handleLogout()}
              >
                {loggingOut ? "Logging out…" : "Log out"}
              </Button>
            </>
          ) : null}
        </header>

        <main className="flex-1 overflow-auto">
          <div className="mx-auto max-w-5xl px-10 py-10">{children}</div>
        </main>
      </div>
    </div>
  );
}

import {
  MOCK_ADMIN,
  MOCK_ANALYTICS_WEEKLY,
  MOCK_DASHBOARD_STATS,
  MOCK_DRIVER,
  MOCK_RECENT_EVENTS,
  MOCK_SETTINGS,
} from "@/lib/mock-data";
import type {
  AdminDriver,
  AdminDriverDetail,
  AnalyticsData,
  AnalyticsSummary,
  ApiUser,
  DashboardStat,
  FleetSummary,
  RecentEvent,
  SessionInfo,
  SessionLogResponse,
} from "@/lib/analytics-types";
import type { MockSettings } from "@/lib/mock-data";

const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK !== "false";

let mockSessionUser: ApiUser | null = null;

function resolveApiBase(): string {
  const env = process.env.NEXT_PUBLIC_API_BASE;
  if (env === "") {
    return "";
  }
  if (env != null) {
    return env.replace(/\/$/, "");
  }
  return "http://localhost:8000";
}

const API_BASE = resolveApiBase();

/** Next.js `trailingSlash: true` redirects bare paths; keep API calls on the slash URL. */
function withTrailingSlash(path: string): string {
  return path.endsWith("/") ? path : `${path}/`;
}

export type ApiError = {
  message: string;
  status?: number;
};

function summaryFromMockWeekly(): AnalyticsSummary {
  return {
    session_count: MOCK_ANALYTICS_WEEKLY.reduce((n, row) => n + row.sessions, 0),
    total_drowsy_events: MOCK_ANALYTICS_WEEKLY.reduce((n, row) => n + row.alerts, 0),
    total_yawns: 2,
    last_session_duration_s: 1840,
  };
}

function dashboardFromSummary(summary: AnalyticsSummary): DashboardStat[] {
  const duration =
    summary.last_session_duration_s != null
      ? `${Math.round(summary.last_session_duration_s / 60)} min`
      : "—";

  return [
    {
      id: "sessions",
      label: "Total sessions",
      value: String(summary.session_count),
      detail: "Recorded on this account",
    },
    {
      id: "drowsy",
      label: "Drowsiness events",
      value: String(summary.total_drowsy_events),
      detail: "Across all sessions",
    },
    {
      id: "yawns",
      label: "Yawn events",
      value: String(summary.total_yawns),
      detail: "Across all sessions",
    },
    {
      id: "duration",
      label: "Last session",
      value: duration,
      detail: "Most recent completed run",
    },
  ];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${withTrailingSlash(path)}`, {
      credentials: "include",
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
  } catch {
    throw {
      message:
        "Failed to fetch — open the app at http://localhost:3000 and ensure the API is running on port 8000.",
      status: 0,
    } satisfies ApiError;
  }

  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string; message?: string };
      message = body.detail ?? body.message ?? message;
    } catch {
      message = await res.text().catch(() => message);
    }
    throw { message, status: res.status } satisfies ApiError;
  }

  if (res.status === 204) {
    return undefined as T;
  }

  return res.json() as Promise<T>;
}

export const api = {
  async getSession() {
    if (USE_MOCK) {
      return { user: (mockSessionUser ?? MOCK_DRIVER) as ApiUser };
    }
    const user = await request<ApiUser>("/api/auth/me");
    return { user };
  },

  async login(name: string, password: string) {
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 400));
      if (!name || !password) {
        throw { message: "Username and password are required" } satisfies ApiError;
      }
      if (name.toLowerCase() === "admin") {
        mockSessionUser = { ...MOCK_ADMIN, name };
        return { user: mockSessionUser };
      }
      mockSessionUser = { ...MOCK_DRIVER, name, driver_id: 1 };
      return { user: mockSessionUser };
    }
    const data = await request<{ user: ApiUser }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ name, password }),
    });
    return data;
  },

  async register(
    name: string,
    password: string,
    contact = "",
    licenseNumber = "",
  ) {
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 500));
      if (!name || !password) {
        throw { message: "All fields are required" } satisfies ApiError;
      }
      return { user: { ...MOCK_DRIVER, name, contact_info: contact } as ApiUser };
    }
    await request<{ message: string }>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({
        name,
        password,
        contact,
        license_number: licenseNumber,
      }),
    });
    return this.login(name, password);
  },

  async logout() {
    if (USE_MOCK) {
      mockSessionUser = null;
      return { message: "ok" };
    }
    return request<{ message: string }>("/api/auth/logout", { method: "POST" });
  },

  async getAnalytics(): Promise<AnalyticsData> {
    if (USE_MOCK) {
      return {
        weekly: [...MOCK_ANALYTICS_WEEKLY],
        summary: summaryFromMockWeekly(),
      };
    }
    const summary = await request<AnalyticsSummary>("/api/analytics/summary");
    return {
      weekly: [],
      summary,
    };
  },

  async getDashboardStats(): Promise<DashboardStat[]> {
    if (USE_MOCK) {
      return [...MOCK_DASHBOARD_STATS];
    }
    const { summary } = await this.getAnalytics();
    return dashboardFromSummary(summary);
  },

  async getRecentEvents(): Promise<RecentEvent[]> {
    if (USE_MOCK) {
      return [...MOCK_RECENT_EVENTS];
    }
    const { sessions } = await request<{ sessions: SessionInfo[] }>("/api/sessions");
    return sessions.slice(0, 8).map((session) => ({
      id: session.session_id,
      time: session.session_id.replace(/_/g, " "),
      type: session.active ? "active" : "session",
      message: session.active ? "Monitoring session in progress" : "Session recorded",
    }));
  },

  async getSettings(): Promise<MockSettings> {
    if (USE_MOCK) {
      return MOCK_SETTINGS;
    }
    return MOCK_SETTINGS;
  },

  async startSession(earThreshold = 0.25): Promise<{ session_id: string }> {
    if (USE_MOCK) {
      return { session_id: "mock_session" };
    }
    return request<{ session_id: string }>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ ear_threshold: earThreshold }),
    });
  },

  async stopSession(sessionId: string): Promise<void> {
    if (USE_MOCK) return;
    await request(`/api/sessions/${encodeURIComponent(sessionId)}/stop`, {
      method: "POST",
    });
  },

  admin: {
    async fleetSummary() {
      if (USE_MOCK) {
        return {
          driver_count: 3,
          session_count: 12,
          total_drowsy_events: 7,
          total_yawns: 4,
        };
      }
      return request<FleetSummary>("/api/admin/fleet/summary");
    },

    async listDrivers() {
      if (USE_MOCK) {
        return {
          drivers: [
            {
              driver_id: 1,
              name: "Alex Morgan",
              contact_info: "alex@example.com",
              license_number: "DL-1001",
              session_count: 5,
            },
            {
              driver_id: 2,
              name: "Jordan Lee",
              contact_info: "",
              license_number: "DL-1002",
              session_count: 3,
            },
          ],
        };
      }
      return request<{ drivers: AdminDriver[] }>("/api/admin/drivers");
    },

    async getDriver(driverId: number) {
      if (USE_MOCK) {
        return {
          driver_id: driverId,
          name: "Alex Morgan",
          contact_info: "alex@example.com",
          license_number: "DL-1001",
          session_count: 5,
          total_drowsy_events: 2,
          total_yawns: 1,
          last_session_duration_s: 1200,
        };
      }
      return request<AdminDriverDetail>(`/api/admin/drivers/${driverId}`);
    },

    async listSessions(driverId: number) {
      if (USE_MOCK) {
        return {
          sessions: [
            { session_id: "20260101_120000", log_file: null, active: false },
            { session_id: "20260102_090000", log_file: null, active: false },
          ],
        };
      }
      return request<{ sessions: SessionInfo[] }>(
        `/api/admin/drivers/${driverId}/sessions`,
      );
    },

    async getSessionLog(driverId: number, sessionId: string, tail?: number) {
      if (USE_MOCK) {
        return {
          session_id: sessionId,
          content: "[2026-01-01 12:00:00] [INFO] SESSION_STARTED\n",
        };
      }
      const qs = tail != null ? `?tail=${tail}` : "";
      return request<SessionLogResponse>(
        `/api/admin/drivers/${driverId}/sessions/${encodeURIComponent(sessionId)}/log${qs}`,
      );
    },
  },
};

export { USE_MOCK, API_BASE };

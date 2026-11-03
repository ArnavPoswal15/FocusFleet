/**
 * Mock data for local UI development. Replace with live API responses in production.
 */

export const MOCK_ADMIN = {
  driver_id: 0,
  name: "Fleet Admin",
  role: "admin",
} as const;

export const MOCK_DRIVER = {
  id: "drv_001",
  name: "Alex Morgan",
  email: "alex.morgan@example.com",
  fleetId: "fleet_north_7",
  role: "driver",
} as const;

export const MOCK_DASHBOARD_STATS = [
  {
    id: "sessions",
    label: "Sessions today",
    value: "12",
    detail: "+2 vs yesterday",
  },
  {
    id: "alert",
    label: "Drowsiness alerts",
    value: "3",
    detail: "Last at 09:41",
  },
  {
    id: "focus",
    label: "Avg. focus score",
    value: "87%",
    detail: "Rolling 24h",
  },
  {
    id: "uptime",
    label: "Camera uptime",
    value: "99.2%",
    detail: "This week",
  },
] as const;

export const MOCK_RECENT_EVENTS = [
  {
    id: "evt_1",
    time: "09:41",
    type: "drowsy",
    message: "Eye closure threshold exceeded",
  },
  {
    id: "evt_2",
    time: "08:15",
    type: "ok",
    message: "Session started",
  },
  {
    id: "evt_3",
    time: "07:52",
    type: "warning",
    message: "Head pose deviation detected",
  },
] as const;

export const MOCK_ANALYTICS_WEEKLY = [
  { day: "Mon", sessions: 8, alerts: 1 },
  { day: "Tue", sessions: 10, alerts: 2 },
  { day: "Wed", sessions: 7, alerts: 0 },
  { day: "Thu", sessions: 12, alerts: 3 },
  { day: "Fri", sessions: 9, alerts: 1 },
  { day: "Sat", sessions: 4, alerts: 0 },
  { day: "Sun", sessions: 5, alerts: 1 },
] as const;

export type MockSettings = {
  alertSound: boolean;
  sensitivity: "low" | "medium" | "high";
  autoStartMonitoring: boolean;
  dataRetentionDays: number;
};

export const MOCK_SETTINGS: MockSettings = {
  alertSound: true,
  sensitivity: "medium",
  autoStartMonitoring: false,
  dataRetentionDays: 30,
};

export type AnalyticsDayRow = {
  day: string;
  sessions: number;
  alerts: number;
};

export type AnalyticsSummary = {
  session_count: number;
  total_drowsy_events: number;
  total_yawns: number;
  last_session_duration_s: number | null;
};

export type AnalyticsData = {
  weekly: AnalyticsDayRow[];
  summary: AnalyticsSummary;
};

export type DashboardStat = {
  id: string;
  label: string;
  value: string;
  detail: string;
};

export type RecentEvent = {
  id: string;
  time: string;
  type: string;
  message: string;
};

export type ApiUser = {
  driver_id?: number;
  name: string;
  role?: "driver" | "admin";
  email?: string;
  contact_info?: string;
  license_number?: string;
};

export type AdminDriver = {
  driver_id: number;
  name: string;
  contact_info?: string | null;
  license_number?: string | null;
  session_count: number;
};

export type AdminDriverDetail = AdminDriver & AnalyticsSummary;

export type FleetSummary = {
  driver_count: number;
  session_count: number;
  total_drowsy_events: number;
  total_yawns: number;
};

export type SessionLogResponse = {
  session_id: string;
  content: string;
  truncated?: boolean;
};

export type SessionInfo = {
  session_id: string;
  log_file: string | null;
  active: boolean;
};

import { SessionLogView } from "./session-log-view";

export function generateStaticParams() {
  return [{ id: "0", sessionId: "00000000_000000" }];
}

export default function AdminSessionLogPage() {
  return <SessionLogView />;
}

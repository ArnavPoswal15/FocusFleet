import { PageHeader } from "@/components/layout/page-header";
import { MonitoringPanel } from "./monitoring-panel";

export default function MonitoringPage() {
  return (
    <>
      <PageHeader
        title="Monitoring"
        description="Live camera preview with drowsiness detection signals."
      />
      <MonitoringPanel />
    </>
  );
}

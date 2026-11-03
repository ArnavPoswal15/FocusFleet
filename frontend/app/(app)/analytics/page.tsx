import { PageHeader } from "@/components/layout/page-header";
import { AnalyticsView } from "./analytics-view";

export default function AnalyticsPage() {
  return (
    <>
      <PageHeader
        title="Analytics"
        description="Session volume and alert trends from your monitoring history."
      />
      <AnalyticsView />
    </>
  );
}

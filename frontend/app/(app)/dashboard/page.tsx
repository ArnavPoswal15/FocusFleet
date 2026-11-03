import { PageHeader } from "@/components/layout/page-header";
import { DashboardView } from "./dashboard-view";

export default function DashboardPage() {
  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Overview of your monitoring activity."
      />
      <DashboardView />
    </>
  );
}

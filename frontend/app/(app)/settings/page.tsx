import { PageHeader } from "@/components/layout/page-header";
import { SettingsView } from "./settings-view";

export default function SettingsPage() {
  return (
    <>
      <PageHeader
        title="Settings"
        description="Alerts, sensitivity, and local preferences."
      />
      <SettingsView />
    </>
  );
}

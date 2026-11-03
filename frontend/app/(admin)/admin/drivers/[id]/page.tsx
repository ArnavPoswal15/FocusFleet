import { DriverDetailView } from "./driver-detail-view";

export function generateStaticParams() {
  return [{ id: "0" }];
}

export default function AdminDriverDetailPage() {
  return <DriverDetailView />;
}

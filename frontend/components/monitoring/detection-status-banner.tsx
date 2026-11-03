"use client";

import { cn } from "@/lib/utils";

export type DetectionBannerVariant =
  | "alert"
  | "warning"
  | "drowsy"
  | "no_face"
  | "neutral"
  | "unknown";

const COPY: Record<DetectionBannerVariant, string> = {
  alert: "Alert — driver is focused",
  warning: "Monitoring — sustained low alertness",
  drowsy: "Drowsiness detected — stay alert",
  no_face: "No face detected",
  neutral: "Waiting for camera…",
  unknown: "Connecting detection…",
};

const STYLES: Record<DetectionBannerVariant, string> = {
  alert: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  warning: "border-amber-500/30 bg-amber-500/10 text-amber-200",
  drowsy: "border-red-500/40 bg-red-500/15 text-red-200",
  no_face: "border-border bg-white/[0.03] text-muted-foreground",
  neutral: "border-border bg-white/[0.03] text-muted-foreground",
  unknown: "border-border bg-white/[0.03] text-muted-foreground",
};

export function mapStatusToBanner(
  driverStatus?: string,
  state?: string,
): DetectionBannerVariant {
  switch (driverStatus) {
    case "alert":
      return "alert";
    case "warning":
      return "warning";
    case "drowsy":
      return "drowsy";
    case "no_face":
      return "no_face";
    case "neutral":
      return "neutral";
    default:
      break;
  }
  switch (state) {
    case "focused":
      return "alert";
    case "distracted":
      return "warning";
    case "drowsy":
      return "drowsy";
    default:
      return "unknown";
  }
}

export function DetectionStatusBanner({
  variant,
  className,
}: {
  variant: DetectionBannerVariant;
  className?: string;
}) {
  return (
    <div
      role="status"
      className={cn(
        "flex items-center gap-2 rounded-xl border px-3 py-2.5 text-[13px] font-medium transition-colors duration-150",
        STYLES[variant],
        variant === "drowsy" && "animate-pulse",
        className,
      )}
    >
      <span
        className={cn(
          "size-2 shrink-0 rounded-full",
          variant === "alert" && "bg-emerald-400",
          variant === "warning" && "bg-amber-400",
          variant === "drowsy" && "bg-red-400",
          (variant === "no_face" || variant === "neutral" || variant === "unknown") &&
            "bg-muted-foreground",
        )}
        aria-hidden
      />
      {COPY[variant]}
    </div>
  );
}

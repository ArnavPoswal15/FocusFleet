import { cn } from "@/lib/utils";

export function PageLoading({ className }: { className?: string }) {
  return (
    <div className={cn("space-y-6 animate-pulse", className)} aria-busy="true">
      <div className="h-4 w-48 rounded-md bg-white/[0.06]" />
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="h-28 rounded-2xl bg-white/[0.04]" />
        <div className="h-28 rounded-2xl bg-white/[0.04]" />
      </div>
      <div className="h-52 rounded-2xl bg-white/[0.04]" />
    </div>
  );
}

export function PageEmpty({
  title,
  description,
}: {
  title: string;
  description?: string;
}) {
  return (
    <div className="rounded-2xl border border-dashed border-border px-6 py-14 text-center">
      <p className="text-sm font-medium text-foreground">{title}</p>
      {description ? (
        <p className="mt-1.5 text-[13px] text-muted-foreground">{description}</p>
      ) : null}
    </div>
  );
}

export function PageError({
  message,
  onRetry,
}: {
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card px-6 py-10 text-center">
      <p className="text-sm font-medium text-foreground">Something went wrong</p>
      <p className="mt-1.5 text-[13px] text-muted-foreground">{message}</p>
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 text-[13px] font-medium text-accent underline-offset-4 hover:underline"
        >
          Try again
        </button>
      ) : null}
    </div>
  );
}

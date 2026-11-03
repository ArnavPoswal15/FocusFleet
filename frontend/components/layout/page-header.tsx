type PageHeaderProps = {
  title: string;
  description?: string;
};

export function PageHeader({ title, description }: PageHeaderProps) {
  return (
    <header className="mb-10 space-y-1.5">
      <h1 className="text-[34px] font-semibold tracking-tight text-foreground">
        {title}
      </h1>
      {description ? (
        <p className="text-[13px] text-muted-foreground">{description}</p>
      ) : null}
    </header>
  );
}

import type { ReactNode } from "react";

interface DashboardShellProps {
  header: ReactNode;
  sidebar: ReactNode;
  viewer: ReactNode;
  infoPanel: ReactNode;
}

export function DashboardShell({ header, sidebar, viewer, infoPanel }: DashboardShellProps) {
  return (
    <div className="dashboard-shell flex h-screen flex-col overflow-hidden">
      {header}
      <div className="flex min-h-0 flex-1">
        {sidebar}
        <div className="flex min-w-0 flex-1">
          {viewer}
          <div className="hidden w-[320px] shrink-0 xl:block">{infoPanel}</div>
        </div>
      </div>
      {/* Info panel stacks below viewer on narrower viewports */}
      <div className="max-h-[40vh] overflow-y-auto border-t border-slate-200 xl:hidden">
        {infoPanel}
      </div>
    </div>
  );
}

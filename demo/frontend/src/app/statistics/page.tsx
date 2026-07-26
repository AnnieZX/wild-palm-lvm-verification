import Link from "next/link";

import { StatisticsDashboard } from "@/components/statistics";
import { MOCK_STATISTICS } from "@/lib/mock/statistics";

export default function StatisticsPage() {
  return (
    <div className="min-h-screen bg-slate-100">
      <nav className="border-b border-slate-200 bg-white px-5 py-3">
        <div className="mx-auto flex max-w-7xl items-center gap-4">
          <Link href="/" className="wp-link">
            ← Sample viewer
          </Link>
          <span className="text-sm text-slate-400">|</span>
          <span className="text-sm font-medium text-slate-700">Statistics dashboard</span>
        </div>
      </nav>

      <main className="mx-auto max-w-7xl px-5 py-8">
        <StatisticsDashboard data={MOCK_STATISTICS} />
      </main>
    </div>
  );
}

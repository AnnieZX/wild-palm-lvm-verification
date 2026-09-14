import { ConfusionMatrixChart } from "@/components/statistics/ConfusionMatrixChart";
import { DecisionDistributionChart } from "@/components/statistics/DecisionDistributionChart";
import { GroupedComparisonChart } from "@/components/statistics/GroupedComparisonChart";
import { MetricBarChart } from "@/components/statistics/MetricBarChart";
import type { StatisticsDashboardData } from "@/components/statistics/types";

export interface StatisticsDashboardProps {
  data: StatisticsDashboardData;
  className?: string;
}

/**
 * Statistics dashboard grid populated from GET /api/v1/statistics.
 */
export function StatisticsDashboard({ data, className = "" }: StatisticsDashboardProps) {
  const modelRows = data.model_comparison.map((row) => ({
    id: row.model_key,
    label: row.display_name,
    precision: row.precision,
    recall: row.recall,
    f1: row.f1,
    accuracy: row.accuracy,
  }));

  const promptRows = data.prompt_comparison.map((row) => ({
    id: row.ablation,
    label: `${row.ablation} · ${row.condition}`,
    precision: row.precision,
    recall: row.recall,
    f1: row.f1,
    accuracy: row.accuracy,
  }));

  return (
    <div className={`space-y-4 ${className}`}>
      <header className="rounded border border-slate-200 bg-white px-4 py-3">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
          Evaluation
        </p>
        <h1 className="mt-0.5 text-lg font-semibold text-slate-900">
          Verification metrics
        </h1>
        <dl className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-sm">
          <div>
            <dt className="text-xs text-slate-500">Experiment</dt>
            <dd className="font-mono text-slate-800">{data.experiment_id}</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500">Model</dt>
            <dd className="font-mono text-slate-800">{data.model_key}</dd>
          </div>
          <div>
            <dt className="text-xs text-slate-500">Evaluated</dt>
            <dd className="font-mono text-slate-800">
              {data.evaluated_samples.toLocaleString()} / {data.dataset_size.toLocaleString()}
            </dd>
          </div>
        </dl>
      </header>

      <MetricBarChart metrics={data.primary_metrics} />

      <div className="grid gap-4 lg:grid-cols-2">
        <DecisionDistributionChart distribution={data.decision_distribution} />
        <ConfusionMatrixChart matrix={data.confusion_matrix} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <GroupedComparisonChart
          title="Model comparison"
          subtitle="F1 across models with available evaluation artifacts"
          rows={modelRows}
          highlightMetric="f1"
        />
        <GroupedComparisonChart
          title="Ablation comparison"
          subtitle="F1 across available ablations for the primary model"
          rows={promptRows}
          highlightMetric="f1"
        />
      </div>
    </div>
  );
}

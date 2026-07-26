import type { GroundTruthLabel, VerificationDecision } from "@shared/types";

/** Ground-truth context shared across model predictions for one sample. */
export interface ComparisonGroundTruth {
  matched: boolean;
  label: GroundTruthLabel;
  maxIou: number;
}

/** One VLM's verification output for a single sample (frontend view model). */
export interface ModelPredictionResult {
  modelKey: string;
  modelName: string;
  decision: VerificationDecision;
  /** Model-reported or derived confidence score (0–1). */
  confidence: number | null;
  reasoning: string;
  visualReasoning?: string;
  confidenceReasoning?: string;
  processingTimeMs: number;
  promptVersion: string;
  groundTruth: ComparisonGroundTruth;
}

/** All model predictions for one sample — input to ModelComparisonPanel. */
export interface SampleComparisonData {
  sampleId: string;
  imageName: string;
  predictions: ModelPredictionResult[];
}

export type Role = "admin" | "insights_user";

export interface User {
  username: string;
  role: Role;
  display_name: string;
}

export type Status = "completed" | "needs_clarification" | "blocked" | "failed";

export interface Filter {
  field: string;
  op: string;
  value: string | number | string[];
  label: string;
}

export interface MlPrediction {
  enabled: boolean;
  prediction?: string | number;
  confidence?: number;
}

export interface Intent {
  raw: string;
  dataset: string;
  ml_intent?: MlPrediction;
  kind: "aggregate" | "ranking" | "list" | "distribution" | "correlation" | "drivers" | "unknown";
  measures: { col: string | null; agg: string | null }[];
  group_by: { col: string; bin: number | null }[];
  filters: { col: string; op: string; value: unknown; label: string }[];
  ranking: { direction: "ASC" | "DESC"; limit: number; target: "groups" | "rows" } | null;
  share: boolean;
  follow_up: boolean;
  notes: string[];
  unknown_terms: string[];
}

export interface DatasetSummary {
  id: string;
  name: string;
  description: string;
  source: string;
  entity: { singular: string; plural: string; id_label: string };
  rows: number;
  columns: number;
  tables: string[];
  primary_measure: string;
}

export interface DriverFactor {
  factor: string;
  label: string;
  spread: number;
  best: { level: string; value: number; n: number };
  worst: { level: string; value: number; n: number };
  levels: { level: string; value: number | null; n: number; supported: boolean }[];
}

export interface ClarificationOption {
  id: string;
  label: string;
  preview?: string | null;
}

export interface Ambiguity {
  ml_classifier?: MlPrediction;
  ambiguity_score: number;
  threshold: number;
  signals: { type: string; weight: number; detail: string }[];
  needs_clarification: boolean;
  question: string | null;
  options: ClarificationOption[];
  comparison_detail?: string;
}

export interface SchemaChunk {
  table: string;
  content: string;
  score: number;
  reason: string;
}

export interface SchemaContext {
  k: number;
  embedder: string;
  candidates: { table: string; score: number }[];
  chunks: SchemaChunk[];
  tables: string[];
  actions: string[];
  total_tables: number;
}

export interface CheckError {
  code: string;
  message: string;
  repairable?: boolean;
}

export interface SemanticQuestion {
  question: string;
  answer: boolean;
  detail: string;
  source: string;
}

export interface ValidationReport {
  passed: boolean;
  hallucination_types: string[];
  timings_ms: Record<string, number>;
  checks: {
    schema: { passed: boolean; errors: CheckError[]; referenced: string[] };
    semantic: { passed: boolean; errors: CheckError[]; questions: SemanticQuestion[]; warnings: string[]; skipped?: boolean };
    structural: {
      passed: boolean;
      errors: CheckError[];
      joins: { condition: string; valid: boolean; foreign_key: string | null }[];
      skipped?: boolean;
    };
    ml_classifier: {
      passed: boolean;
      errors: CheckError[];
      skipped?: boolean;
      enabled?: boolean;
      prediction?: string;
      confidence?: number;
      threshold?: number;
      probabilities?: Record<string, number>;
      informational_only?: boolean;
      reason?: string;
    };
  };
}

export interface Attempt {
  attempt: number;
  sql: string;
  source: string;
  validation: ValidationReport | null;
  repair: { sql: string; blocked: boolean; strategies: string[]; changes: string[] } | null;
}

export interface TraceStep {
  key: string;
  name: string;
  status: "done" | "failed" | "waiting" | "skipped";
  summary: string;
  detail: unknown;
  attempt: number | null;
  duration_ms: number;
}

export interface QueryResult {
  columns: string[];
  rows: (string | number | null)[][];
  row_count: number;
  truncated: boolean;
  shown?: number;
  exec_ms: number;
}

export interface Visualization {
  chart_type: "bar" | "line" | "pie" | "kpi" | "table" | "empty" | "histogram" | "scatter" | "drivers";
  drivers: { outcome: string; unit: string; min_support: number; factors: DriverFactor[] } | null;
  label_column: string | null;
  key_column: string | null;
  points: { x: number; y: number; label: string }[];
  x_label: string | null;
  y_label: string | null;
  stats: { pearson_r: number | null; n: number; plotted: number } | null;
  value_column: string | null;
  value_label: string | null;
  labels: string[];
  values: number[];
  horizontal: boolean;
  explanation: string;
  sql_in_plain_english: string[];
}

export interface ResultCheck {
  name: string;
  title: string;
  applicable: boolean;
  passed: boolean;
  detail: string;
}

export interface DataWarning {
  code: string;
  severity: "high" | "medium" | "low";
  title: string;
  detail: string;
}

export interface Confidence {
  score: number;
  level: "high" | "medium" | "low";
  label: string;
  abstain: boolean;
  reasons: { text: string; impact: number }[];
}

export interface Evidence {
  column_labels: Record<string, string>;
  sql: string;
  columns: string[];
  rows: (string | number | null)[][];
  row_count: number;
  truncated: boolean;
  exec_ms: number;
}

export interface NarrativeSection {
  key: string;
  title: string;
  paragraphs: string[];
  bullets: string[];
  bullets_after: number;
}

export interface PipelineResponse {
  response_id: string;
  dataset_id: string;
  column_labels: Record<string, string>;
  session_id: string;
  status: Status;
  question: string;
  error: string | null;
  notes: string[];
  warnings: string[];
  clarification: Ambiguity | null;
  ambiguity: Ambiguity | null;
  intent: Intent | null;
  schema_context: SchemaContext | null;
  access_scope: { role: Role; denied: { table: string; column: string }[] } | null;
  plan: string[] | null;
  attempts: Attempt[];
  injected: { type: string | null; description: string } | null;
  final_sql: string | null;
  result: QueryResult | null;
  visualization: Visualization | null;
  result_checks: { passed: boolean; checks: ResultCheck[]; applicable: number; failed: string[] } | null;
  data_warnings: DataWarning[];
  confidence: Confidence | null;
  followups: string[];
  auto_resolved: { original: string; chosen: string; compared: string[]; detail: string } | null;
  narrative: { sections: NarrativeSection[] } | null;
  trace: TraceStep[];
  latency_ms: number;
}

export interface SuggestionGroup {
  group: string;
  icon: string;
  questions: string[];
}

export interface SchemaTable {
  name: string;
  description: string;
  columns: { name: string; type: string; primary_key: boolean; role: string; nulls: number; description: string }[];
  foreign_keys: { column: string; references: string }[];
  indexed_at: number | null;
  chunk: string | null;
}

export interface AuditEntry {
  id: number;
  ts: number;
  username: string;
  role: Role;
  session_id: string;
  dataset_id: string | null;
  question: string;
  final_sql: string | null;
  status: Status;
  row_count: number | null;
  latency_ms: number;
  repair_attempts: number;
  confidence: number | null;
  hallucinations: { attempt: number; types: string[] }[];
}

export interface AuditResponse {
  entries: AuditEntry[];
  stats: {
    total: number;
    completed: number;
    hallucinations_detected: number;
    repaired: number;
    blocked: number;
    p80_latency_ms: number | null;
  };
}

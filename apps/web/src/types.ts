export type Language = "zh" | "en";

export interface RunIndexEntry {
  run_id: string;
  label: string;
  path: string;
}

export interface RunManifest {
  run_id: string;
  label: string;
  is_demo?: boolean;
  model: string;
  status: string;
  created_at: string;
  description: string;
  jlens_layer: number;
  patch_layers: number[];
  languages: Language[];
  categories: string[];
  artifact_version: number;
}

export interface TokenScore {
  rank: number;
  token: string;
  score: number;
  is_expected?: boolean;
}

export interface ReadoutRecord {
  run_id: string;
  pair_id: string;
  language: Language;
  category: string;
  concept: string;
  abstractness: "abstract" | "concrete";
  prompt: string;
  expected: string[];
  jlens_layer: number;
  position: number;
  topk: TokenScore[];
  concept_hit: boolean;
}

export interface PatchRecord {
  run_id: string;
  patch_id: string;
  pair_id: string;
  source_language: Language;
  target_language: Language;
  source_concept: string;
  target_context_concept: string;
  category: string;
  abstractness: "abstract" | "concrete";
  patch_layer: number;
  jlens_layer: number;
  control_type: "cross_language_concept" | "same_language_concept" | "random_same_norm" | "unrelated_concept";
  transport_success: boolean;
  language_preserved: boolean;
  concept_score: number;
  source_language_leakage: number;
  null_gap: number;
  readout: TokenScore[];
  next_token: TokenScore[];
}

export interface LayerScanRow {
  layer: number;
  cross_language_success: number;
  concrete_success: number;
  same_language_success: number;
  null_success: number;
  source_language_leakage: number;
}

export interface LayerScan {
  run_id: string;
  model: string;
  metric: string;
  rows: LayerScanRow[];
}

export interface ProjectionPoint {
  id: string;
  pair_id: string;
  language: Language;
  category: string;
  concept: string;
  abstractness: "abstract" | "concrete";
  condition: "baseline" | "patched";
  x: number;
  y: number;
}

export interface ProjectionLink {
  source: string;
  target: string;
  pair_id: string;
  kind: string;
}

export interface Projections {
  run_id: string;
  basis: string;
  description: string;
  points: ProjectionPoint[];
  links: ProjectionLink[];
}

export interface Metrics {
  run_id: string;
  primary_patch_layer: number;
  summary: Record<string, number>;
}

export interface RunData {
  manifest: RunManifest;
  readouts: ReadoutRecord[];
  patches: PatchRecord[];
  layerScan: LayerScan;
  projections: Projections;
  metrics: Metrics;
}

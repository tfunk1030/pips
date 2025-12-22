export type RegionConstraint =
  | { type: 'sum'; value: number; size?: number }
  | { type: 'op'; op: '=' | '<' | '>' | '≠'; value: number; size?: number }
  | { type: 'all_equal'; size?: number };

export interface PuzzleSpec {
  name?: string;
  rows: number;
  cols: number;
  regions: number[][]; // region id per cell
  regionConstraints: Record<number, RegionConstraint>;
  maxPip?: number;
  allowDuplicates?: boolean;
}

export interface DominoPlacement {
  id: string;
  cells: [number, number][]; // two cell coordinates [r,c]
  values: [number, number];
}

export interface SolutionGrid {
  gridPips: number[][];
  dominoes: DominoPlacement[];
}

export interface SolverStats {
  nodes: number;
  backtracks: number;
  prunes: number;
  elapsedMs: number;
}

export interface SolveResult {
  status: 'solved' | 'unsat' | 'incomplete';
  solution?: SolutionGrid;
  stats: SolverStats;
  explanation?: string;
  validationReport: ValidationReport;
}

export interface ValidationIssue {
  level: 'info' | 'warning' | 'error';
  message: string;
}

export interface ValidationReport {
  ok: boolean;
  issues: ValidationIssue[];
}

export interface SolverConfig {
  allowDuplicates: boolean;
  maxPip: number;
  findAll: boolean;
  progressInterval: number;
}

export interface SolverProgress {
  nodes: number;
  backtracks: number;
}

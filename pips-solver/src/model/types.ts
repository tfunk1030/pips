/**
 * Core type definitions for the Pips puzzle solver
 */

// ===== Puzzle Specification Types =====

export type ConstraintOp = '=' | '<' | '>' | '≠';

export interface RegionConstraint {
  sum?: number;
  op?: ConstraintOp;
  value?: number;
  all_equal?: boolean;
  size?: number; // Optional sanity check
}

export interface PuzzleSpec {
  id?: string;
  name?: string;
  rows: number;
  cols: number;
  maxPip?: number; // Default 6 (double-six)
  allowDuplicates?: boolean; // Default false
  regions: number[][]; // regions[r][c] = regionId
  constraints: { [regionId: number]: RegionConstraint };
}

// ===== Normalized Puzzle Types =====

export interface Cell {
  row: number;
  col: number;
}

export interface Edge {
  cell1: Cell;
  cell2: Cell;
}

export interface NormalizedPuzzle {
  spec: PuzzleSpec;
  adjacencyList: Map<string, Cell[]>; // cellKey -> neighbors
  regionCells: Map<number, Cell[]>; // regionId -> cells
  edges: Edge[]; // All possible domino edges
}

// ===== Domino Types =====

export interface Domino {
  id: string; // e.g., "0-3", "6-6"
  pip1: number;
  pip2: number;
}

export interface DominoPlacement {
  domino: Domino;
  cell1: Cell;
  cell2: Cell;
}

// ===== Solution Types =====

export interface Solution {
  gridPips: number[][]; // gridPips[r][c] = pip value
  dominoes: DominoPlacement[];
  stats: SolverStats;
}

export interface SolverStats {
  nodes: number;
  backtracks: number;
  prunes: number;
  timeMs: number;
  solutions: number;
}

// ===== Validation Types =====

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  regionChecks?: RegionCheckResult[];
  dominoChecks?: DominoCheckResult[];
}

export interface RegionCheckResult {
  regionId: number;
  valid: boolean;
  constraint: RegionConstraint;
  actualValues: number[];
  message: string;
}

export interface DominoCheckResult {
  valid: boolean;
  message: string;
  domino?: DominoPlacement;
}

// ===== Solver Types =====

export interface SolverState {
  gridPips: (number | null)[][]; // null = unassigned
  usedDominoes: Set<string>; // domino ids
  placements: DominoPlacement[];
  domains: Map<string, number[]>; // cellKey -> possible pip values
}

export interface SolverConfig {
  maxPip: number;
  allowDuplicates: boolean;
  findAll: boolean; // Find all solutions vs first solution
  maxIterationsPerTick: number; // For non-blocking solver
  debugLevel: 0 | 1 | 2; // 0=off, 1=basic, 2=verbose
}

export interface SolverProgress {
  nodes: number;
  backtracks: number;
  prunes: number;
  currentDepth: number;
  completed: boolean;
  cancelled: boolean;
}

// ===== Explanation Types =====

export interface Explanation {
  type: 'unsat' | 'conflict' | 'success';
  message: string;
  details: string[];
  conflicts?: Conflict[];
}

export interface Conflict {
  type: 'region_impossible' | 'domino_exhausted' | 'no_valid_placement';
  description: string;
  affectedCells?: Cell[];
  affectedRegion?: number;
}

// ===== Storage Types =====

export interface StoredPuzzle {
  id: string;
  name: string;
  yaml: string;
  spec: PuzzleSpec;
  createdAt: number;
  updatedAt: number;
  solved?: boolean;
  solution?: Solution;
}

// ===== Settings Types =====

export interface AppSettings {
  defaultMaxPip: number;
  defaultAllowDuplicates: boolean;
  defaultFindAll: boolean;
  defaultDebugLevel: 0 | 1 | 2;
  maxIterationsPerTick: number;
}

// ===== Utility Types =====

export const cellKey = (cell: Cell): string => `${cell.row},${cell.col}`;

export const parseCellKey = (key: string): Cell => {
  const [row, col] = key.split(',').map(Number);
  return { row, col };
};

export const dominoId = (pip1: number, pip2: number): string => {
  const [min, max] = pip1 <= pip2 ? [pip1, pip2] : [pip2, pip1];
  return `${min}-${max}`;
};

export const cellsEqual = (c1: Cell, c2: Cell): boolean => {
  return c1.row === c2.row && c1.col === c2.col;
};

export const isAdjacent = (c1: Cell, c2: Cell): boolean => {
  const rowDiff = Math.abs(c1.row - c2.row);
  const colDiff = Math.abs(c1.col - c2.col);
  return (rowDiff === 1 && colDiff === 0) || (rowDiff === 0 && colDiff === 1);
};

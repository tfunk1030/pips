export type Op = '=' | '<' | '>' | '≠';

export type RegionConstraint = {
  /** Optional sanity check for region size */
  size?: number;
  /** If present, region sum must satisfy op/value. If `sum` present, treated as `op="=" value=sum`. */
  sum?: number;
  op?: Op;
  value?: number;
  /** If true, all cells in the region must have the same pip value. */
  all_equal?: boolean;
};

export type RawPuzzle =
  | RawPuzzleFormalV1
  | RawPuzzleLegacyAsciiV0;

/**
 * Formal schema (recommended).
 *
 * Supports:
 * - grid.rows, grid.cols
 * - optional grid.cells as ASCII mask ('.' exists, '#' missing)
 * - regions as:
 *   - mapping: { "r,c": "A" }
 *   - 2D grid: array of arrays of regionIds
 *   - ASCII grid: multiline string with region labels (use '#' for missing)
 */
export type RawPuzzleFormalV1 = {
  grid: {
    rows: number;
    cols: number;
    cells?: string | string[];
  };
  regions:
    | Record<string, string>
    | string
    | string[]
    | string[][]
    | Array<Array<string | number>>;
  regionConstraints?: Record<string, RegionConstraint>;
  dominoes?: {
    maxPip?: number;
    unique?: boolean;
    tiles?: Array<[number, number]>;
  };
};

/**
 * Legacy ASCII schema (compatible with the Python prototype in this repo).
 */
export type RawPuzzleLegacyAsciiV0 = {
  pips?: { pip_min?: number; pip_max?: number };
  dominoes?: { unique?: boolean; tiles?: Array<[number, number]> };
  board: {
    shape: string | string[];
    regions: string | string[];
  };
  region_constraints?: Record<string, { type: 'sum' | 'all_equal'; op?: '==' | '!=' | '<' | '>'; value?: number }>;
};

export type Cell = {
  id: number;
  r: number;
  c: number;
  regionId: string;
};

export type Region = {
  id: string;
  cellIds: number[];
  constraint: RegionConstraint;
};

export type DominoType = {
  key: string; // `${a},${b}` with a<=b
  a: number;
  b: number;
  count: number; // remaining count is tracked in solver; Infinity allowed
};

export type Puzzle = {
  rows: number;
  cols: number;
  maxPip: number;
  allowDuplicates: boolean;

  // Grid topology
  rcToCellId: number[]; // length rows*cols, -1 for missing
  cells: Cell[]; // only existing
  neighbors: number[][]; // by cellId

  // Regions
  regionIds: string[];
  regionIndexById: Record<string, number>;
  regionByIndex: Region[];
  regionIndexByCellId: number[];

  // Domino domain
  dominoTypes: DominoType[];
};

export type SpecError = { path: string; message: string };

export type SpecReport = {
  ok: boolean;
  errors: SpecError[];
  warnings: SpecError[];
};

export type Solution = {
  gridPips: number[][]; // rows x cols; -1 where missing
  mateCellIdByCellId: number[]; // length = numCells, mate cellId
  dominoes: Array<{
    cellA: { r: number; c: number };
    cellB: { r: number; c: number };
    pipsA: number;
    pipsB: number;
    dominoKey: string;
  }>;
};




/**
 * Grid Validator Module
 * Provides cell-by-cell comparison logic for multi-model extraction results.
 *
 * This module enables the extraction comparison/diff view feature by:
 * 1. Detecting disagreements between model outputs at the cell level
 * 2. Classifying disagreements by severity (critical, warning, info)
 * 3. Aggregating results for display in the comparison UI
 */

import type {
  BoardExtractionResult,
  DominoExtractionResult,
  BoardModelResponse,
  DominoModelResponse,
  DominoPair,
  ConstraintDef,
  ConstraintType,
  ConstraintOp,
} from '../../../model/overlayTypes';

// ════════════════════════════════════════════════════════════════════════════
// Severity Types
// ════════════════════════════════════════════════════════════════════════════

/**
 * Disagreement severity levels for visual highlighting and prioritization
 *
 * - critical: Affects puzzle solvability (must be resolved)
 * - warning: May affect solution quality (should be reviewed)
 * - info: Cosmetic differences (optional to review)
 */
export type DisagreementSeverity = 'critical' | 'warning' | 'info';

// ════════════════════════════════════════════════════════════════════════════
// Cell Detection Types
// ════════════════════════════════════════════════════════════════════════════

/**
 * Cell coordinates within the puzzle grid
 */
export interface CellCoordinate {
  /** Row index (0-based) */
  row: number;
  /** Column index (0-based) */
  col: number;
}

/**
 * Detection data for a single cell from one model's extraction
 *
 * Captures all relevant per-cell information needed for comparison:
 * - Whether the cell is a hole (missing)
 * - Which region the cell belongs to
 * - What constraint applies to that region
 */
export interface CellDetection {
  /** Cell position in the grid */
  coordinate: CellCoordinate;

  /** Whether this cell is a hole (true) or exists (false) */
  isHole: boolean;

  /** Region identifier (e.g., 'A', 'B', 'C') - null if hole */
  region: string | null;

  /** Constraint applied to this cell's region - null if no constraint */
  constraint: ConstraintDef | null;

  /** Model that produced this detection */
  model: string;

  /** Confidence score for this detection (0-1) */
  confidence?: number;
}

/**
 * Per-model detection data for a specific cell
 * Maps model name to that model's detection for the cell
 */
export type CellDetectionsByModel = Record<string, CellDetection>;

// ════════════════════════════════════════════════════════════════════════════
// Disagreement Types
// ════════════════════════════════════════════════════════════════════════════

/**
 * Base disagreement information shared by all disagreement types
 */
export interface BaseDisagreement {
  /** Unique identifier for this disagreement */
  id: string;

  /** Category of disagreement */
  type: DisagreementType;

  /** Severity level for prioritization and visual styling */
  severity: DisagreementSeverity;

  /** Human-readable description of the disagreement */
  description: string;

  /** Map of model name to that model's value for the disagreed field */
  values: Record<string, unknown>;
}

/**
 * Types of disagreements that can occur between model extractions
 */
export type DisagreementType =
  | 'grid_dimensions'
  | 'hole_position'
  | 'region_assignment'
  | 'constraint_type'
  | 'constraint_value'
  | 'constraint_operator'
  | 'domino_count'
  | 'domino_value';

/**
 * Disagreement about a specific cell in the grid
 *
 * Used for hole positions and region assignments where
 * the disagreement is localized to a cell coordinate
 */
export interface CellDisagreement extends BaseDisagreement {
  type: 'hole_position' | 'region_assignment';

  /** Cell location where disagreement occurs */
  coordinate: CellCoordinate;

  /** Per-model detections for this cell */
  detections: CellDetectionsByModel;
}

/**
 * Disagreement about grid dimensions (rows or columns)
 */
export interface GridDimensionDisagreement extends BaseDisagreement {
  type: 'grid_dimensions';

  /** Dimension that disagrees: 'rows' or 'cols' */
  dimension: 'rows' | 'cols';

  /** Per-model dimension values */
  values: Record<string, number>;
}

/**
 * Disagreement about a constraint (type, value, or operator)
 */
export interface ConstraintDisagreement extends BaseDisagreement {
  type: 'constraint_type' | 'constraint_value' | 'constraint_operator';

  /** Region identifier where constraint disagrees */
  region: string;

  /** Per-model constraint definitions */
  values: Record<string, ConstraintDef | null>;
}

/**
 * Disagreement about dominoes (count or individual domino values)
 */
export interface DominoDisagreement extends BaseDisagreement {
  type: 'domino_count' | 'domino_value';

  /** For domino_value: index of the disagreed domino in the list */
  dominoIndex?: number;

  /** Per-model domino values (boolean indicates presence/absence for domino_value type) */
  values: Record<string, DominoPair[] | DominoPair | number | boolean>;
}

/**
 * Union type of all possible disagreements
 */
export type Disagreement =
  | CellDisagreement
  | GridDimensionDisagreement
  | ConstraintDisagreement
  | DominoDisagreement;

// ════════════════════════════════════════════════════════════════════════════
// Comparison Result Types
// ════════════════════════════════════════════════════════════════════════════

/**
 * Summary counts of disagreements by severity
 */
export interface DisagreementSummary {
  /** Total count of all disagreements */
  total: number;

  /** Count of critical disagreements (blocks solving) */
  critical: number;

  /** Count of warning disagreements (may affect solution) */
  warning: number;

  /** Count of info disagreements (cosmetic) */
  info: number;
}

/**
 * Summary counts of disagreements by type
 */
export interface DisagreementsByType {
  /** Grid dimension mismatches (rows, cols) */
  gridDimensions: Disagreement[];

  /** Hole position disagreements */
  holePositions: CellDisagreement[];

  /** Region assignment disagreements */
  regionAssignments: CellDisagreement[];

  /** Constraint disagreements (type, value, operator) */
  constraints: ConstraintDisagreement[];

  /** Domino disagreements (count, values) */
  dominoes: DominoDisagreement[];
}

/**
 * Per-model extraction data normalized for comparison
 */
export interface NormalizedModelResult {
  /** Model identifier */
  model: string;

  /** Whether extraction was successful for this model */
  success: boolean;

  /** Grid dimensions */
  dimensions?: {
    rows: number;
    cols: number;
  };

  /** Parsed shape as 2D boolean array (true = hole) */
  holes?: boolean[][];

  /** Parsed regions as 2D string array (region labels) */
  regions?: (string | null)[][];

  /** Parsed constraints by region label */
  constraints?: Record<string, ConstraintDef>;

  /** Parsed dominoes */
  dominoes?: DominoPair[];

  /** Confidence scores */
  confidence?: {
    board?: number;
    dominoes?: number;
  };

  /** Parse error if extraction failed */
  error?: string;
}

/**
 * Complete result of comparing multiple model extractions
 *
 * Provides all information needed for the comparison UI:
 * - Summary statistics for the header/badge
 * - Categorized disagreements for the list view
 * - Per-model normalized data for side-by-side view
 * - Per-cell disagreement map for grid highlighting
 */
export interface ComparisonResult {
  /** Summary counts by severity */
  summary: DisagreementSummary;

  /** All disagreements grouped by type */
  disagreementsByType: DisagreementsByType;

  /** Flat list of all disagreements (for iteration) */
  allDisagreements: Disagreement[];

  /** Quick lookup: does this cell have any disagreement? */
  cellDisagreementMap: Map<string, CellDisagreement[]>;

  /** Normalized extraction results per model for display */
  modelResults: NormalizedModelResult[];

  /** Models that participated in the comparison */
  modelsCompared: string[];

  /** Model selected as the "consensus" or "best" result */
  selectedModel?: string;

  /** Whether all models agree (no disagreements) */
  isUnanimous: boolean;

  /** Timestamp of when comparison was computed */
  comparedAt: number;
}

// ════════════════════════════════════════════════════════════════════════════
// Helper Types
// ════════════════════════════════════════════════════════════════════════════

/**
 * Input for comparison functions - board model responses
 */
export type BoardComparisonInput = Pick<
  BoardModelResponse,
  'model' | 'parsedData' | 'parseSuccess' | 'confidence'
>[];

/**
 * Input for comparison functions - domino model responses
 */
export type DominoComparisonInput = Pick<
  DominoModelResponse,
  'model' | 'parsedData' | 'parseSuccess' | 'confidence'
>[];

/**
 * Options for the comparison function
 */
export interface ComparisonOptions {
  /** Model to use as reference for ordering differences (optional) */
  referenceModel?: string;

  /** Whether to include info-level disagreements (default: true) */
  includeInfoLevel?: boolean;
}

// ════════════════════════════════════════════════════════════════════════════
// Utility Functions
// ════════════════════════════════════════════════════════════════════════════

/**
 * Generate a unique cell key for Map lookups
 */
export function cellKey(row: number, col: number): string {
  return `${row},${col}`;
}

/**
 * Parse cell key back to coordinates
 */
export function parseCellKey(key: string): CellCoordinate {
  const [row, col] = key.split(',').map(Number);
  return { row, col };
}

/**
 * Create empty comparison result (no disagreements)
 */
export function createEmptyComparisonResult(modelsCompared: string[]): ComparisonResult {
  return {
    summary: { total: 0, critical: 0, warning: 0, info: 0 },
    disagreementsByType: {
      gridDimensions: [],
      holePositions: [],
      regionAssignments: [],
      constraints: [],
      dominoes: [],
    },
    allDisagreements: [],
    cellDisagreementMap: new Map(),
    modelResults: [],
    modelsCompared,
    isUnanimous: true,
    comparedAt: Date.now(),
  };
}

/**
 * Generate unique ID for a disagreement
 */
export function generateDisagreementId(type: DisagreementType, ...identifiers: (string | number)[]): string {
  return `${type}:${identifiers.join(':')}`;
}

// ════════════════════════════════════════════════════════════════════════════
// Parsing and Normalization
// ════════════════════════════════════════════════════════════════════════════

/**
 * Normalize a grid string (shape or regions) to a 2D array
 * Handles both actual newlines and escaped \\n sequences
 */
function parseGridString(gridStr: string, rows: number, cols: number): string[][] {
  // Normalize line endings
  const normalized = gridStr.replace(/\\n/g, '\n').trim();
  const lines = normalized.split('\n');

  const grid: string[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: string[] = [];
    const line = lines[r] || '';
    for (let c = 0; c < cols; c++) {
      row.push(line[c] || '.');
    }
    grid.push(row);
  }

  return grid;
}

/**
 * Parse shape string into 2D boolean holes array
 * '#' = hole (true), '.' = cell (false)
 */
function parseShapeToHoles(shape: string, rows: number, cols: number): boolean[][] {
  const grid = parseGridString(shape, rows, cols);
  return grid.map(row => row.map(cell => cell === '#'));
}

/**
 * Parse regions string into 2D region labels array
 * '#' = hole (null), otherwise = region label
 */
function parseRegions(regions: string, rows: number, cols: number): (string | null)[][] {
  const grid = parseGridString(regions, rows, cols);
  return grid.map(row => row.map(cell => (cell === '#' ? null : cell)));
}

/**
 * Valid constraint types for type checking
 */
const VALID_CONSTRAINT_TYPES: ConstraintType[] = ['none', 'sum', 'all_equal', 'all_different'];
const VALID_CONSTRAINT_OPS: ConstraintOp[] = ['==', '<', '>', '!='];

/**
 * Convert raw constraint data from BoardExtractionResult to typed ConstraintDef
 * Safely handles string-typed fields from model responses
 */
function convertToConstraintDef(
  raw: { type: string; op?: string; value?: number }
): ConstraintDef {
  // Normalize type string to valid ConstraintType
  const normalizedType = raw.type.toLowerCase().replace(/[_\s]/g, '') as string;
  let constraintType: ConstraintType = 'none';

  if (normalizedType === 'sum' || normalizedType === 'total') {
    constraintType = 'sum';
  } else if (normalizedType === 'allequal' || normalizedType === 'equal') {
    constraintType = 'all_equal';
  } else if (normalizedType === 'alldifferent' || normalizedType === 'different' || normalizedType === 'unique') {
    constraintType = 'all_different';
  } else if (VALID_CONSTRAINT_TYPES.includes(raw.type as ConstraintType)) {
    constraintType = raw.type as ConstraintType;
  }

  // Build result
  const result: ConstraintDef = { type: constraintType };

  if (raw.op && VALID_CONSTRAINT_OPS.includes(raw.op as ConstraintOp)) {
    result.op = raw.op as ConstraintOp;
  }

  if (raw.value !== undefined) {
    result.value = raw.value;
  }

  return result;
}

/**
 * Convert all constraints from BoardExtractionResult to typed Record<string, ConstraintDef>
 */
function convertConstraints(
  raw: Record<string, { type: string; op?: string; value?: number }> | undefined
): Record<string, ConstraintDef> | undefined {
  if (!raw) return undefined;

  const result: Record<string, ConstraintDef> = {};
  for (const [region, constraint] of Object.entries(raw)) {
    result[region] = convertToConstraintDef(constraint);
  }
  return result;
}

/**
 * Normalize a single BoardModelResponse into a NormalizedModelResult
 */
function normalizeBoardResponse(response: BoardModelResponse): NormalizedModelResult {
  if (!response.parseSuccess || !response.parsedData) {
    return {
      model: response.model,
      success: false,
      error: response.parseError || 'Parse failed',
    };
  }

  const data = response.parsedData;
  const rows = data.rows;
  const cols = data.cols;

  return {
    model: response.model,
    success: true,
    dimensions: { rows, cols },
    holes: parseShapeToHoles(data.shape, rows, cols),
    regions: parseRegions(data.regions, rows, cols),
    constraints: convertConstraints(data.constraints),
    confidence: {
      board: data.confidence
        ? (data.confidence.grid + data.confidence.regions + data.confidence.constraints) / 3
        : undefined,
    },
  };
}

/**
 * Normalize a single DominoModelResponse into domino data
 */
function normalizeDominoResponse(response: DominoModelResponse): Pick<NormalizedModelResult, 'dominoes' | 'confidence'> & { error?: string } {
  if (!response.parseSuccess || !response.parsedData) {
    return {
      error: response.parseError || 'Parse failed',
    };
  }

  return {
    dominoes: response.parsedData.dominoes,
    confidence: {
      dominoes: response.parsedData.confidence,
    },
  };
}

// ════════════════════════════════════════════════════════════════════════════
// Comparison Logic
// ════════════════════════════════════════════════════════════════════════════

/**
 * Normalize domino to canonical form for comparison
 * Sort pips so [3,1] becomes [1,3] for order-independent matching
 */
function normalizeDomino(domino: DominoPair): string {
  const sorted = [...domino].sort((a, b) => a - b) as DominoPair;
  return `${sorted[0]}-${sorted[1]}`;
}

/**
 * Compare grid dimensions across models
 * Returns dimension disagreements if any model has different rows/cols
 */
function compareDimensions(
  modelResults: NormalizedModelResult[]
): GridDimensionDisagreement[] {
  const disagreements: GridDimensionDisagreement[] = [];
  const successfulResults = modelResults.filter(r => r.success && r.dimensions);

  if (successfulResults.length < 2) return disagreements;

  // Check for row disagreements
  const rowValues: Record<string, number> = {};
  const colValues: Record<string, number> = {};
  let hasRowDisagreement = false;
  let hasColDisagreement = false;

  const firstRows = successfulResults[0].dimensions!.rows;
  const firstCols = successfulResults[0].dimensions!.cols;

  for (const result of successfulResults) {
    const dims = result.dimensions!;
    rowValues[result.model] = dims.rows;
    colValues[result.model] = dims.cols;

    if (dims.rows !== firstRows) hasRowDisagreement = true;
    if (dims.cols !== firstCols) hasColDisagreement = true;
  }

  if (hasRowDisagreement) {
    disagreements.push({
      id: generateDisagreementId('grid_dimensions', 'rows'),
      type: 'grid_dimensions',
      dimension: 'rows',
      severity: 'critical',
      description: `Row count disagreement: ${Object.entries(rowValues).map(([m, v]) => `${m}=${v}`).join(', ')}`,
      values: rowValues,
    });
  }

  if (hasColDisagreement) {
    disagreements.push({
      id: generateDisagreementId('grid_dimensions', 'cols'),
      type: 'grid_dimensions',
      dimension: 'cols',
      severity: 'critical',
      description: `Column count disagreement: ${Object.entries(colValues).map(([m, v]) => `${m}=${v}`).join(', ')}`,
      values: colValues,
    });
  }

  return disagreements;
}

/**
 * Compare hole positions across models (shape comparison)
 * Returns cell disagreements where models disagree on whether a cell is a hole
 */
function compareHolePositions(
  modelResults: NormalizedModelResult[],
  maxRows: number,
  maxCols: number
): CellDisagreement[] {
  const disagreements: CellDisagreement[] = [];
  const successfulResults = modelResults.filter(r => r.success && r.holes);

  if (successfulResults.length < 2) return disagreements;

  // Compare each cell position
  for (let row = 0; row < maxRows; row++) {
    for (let col = 0; col < maxCols; col++) {
      const detections: CellDetectionsByModel = {};
      let hasDisagreement = false;
      let firstValue: boolean | undefined;

      for (const result of successfulResults) {
        // Handle different grid sizes - treat out-of-bounds as hole
        const isHole = row >= (result.dimensions?.rows || 0) ||
                       col >= (result.dimensions?.cols || 0) ||
                       (result.holes?.[row]?.[col] ?? true);

        detections[result.model] = {
          coordinate: { row, col },
          isHole,
          region: isHole ? null : (result.regions?.[row]?.[col] ?? null),
          constraint: null, // Filled in by constraint comparison
          model: result.model,
          confidence: result.confidence?.board,
        };

        if (firstValue === undefined) {
          firstValue = isHole;
        } else if (isHole !== firstValue) {
          hasDisagreement = true;
        }
      }

      if (hasDisagreement) {
        const values: Record<string, boolean> = {};
        for (const [model, detection] of Object.entries(detections)) {
          values[model] = detection.isHole;
        }

        disagreements.push({
          id: generateDisagreementId('hole_position', row, col),
          type: 'hole_position',
          severity: 'critical',
          coordinate: { row, col },
          description: `Hole disagreement at (${row},${col}): ${Object.entries(values).map(([m, v]) => `${m}=${v ? 'hole' : 'cell'}`).join(', ')}`,
          detections,
          values,
        });
      }
    }
  }

  return disagreements;
}

/**
 * Compare region assignments across models
 * Returns cell disagreements where models assign different regions to the same cell
 */
function compareRegionAssignments(
  modelResults: NormalizedModelResult[],
  maxRows: number,
  maxCols: number
): CellDisagreement[] {
  const disagreements: CellDisagreement[] = [];
  const successfulResults = modelResults.filter(r => r.success && r.regions);

  if (successfulResults.length < 2) return disagreements;

  // Compare each cell position
  for (let row = 0; row < maxRows; row++) {
    for (let col = 0; col < maxCols; col++) {
      const detections: CellDetectionsByModel = {};
      let hasDisagreement = false;
      let firstValue: string | null | undefined;
      let allHoles = true;

      for (const result of successfulResults) {
        // Skip out-of-bounds cells
        if (row >= (result.dimensions?.rows || 0) || col >= (result.dimensions?.cols || 0)) {
          continue;
        }

        const isHole = result.holes?.[row]?.[col] ?? false;
        const region = result.regions?.[row]?.[col] ?? null;

        // Skip hole cells - they shouldn't have region assignments
        if (isHole) continue;

        allHoles = false;

        detections[result.model] = {
          coordinate: { row, col },
          isHole,
          region,
          constraint: null,
          model: result.model,
          confidence: result.confidence?.board,
        };

        if (firstValue === undefined) {
          firstValue = region;
        } else if (region !== firstValue) {
          hasDisagreement = true;
        }
      }

      // Only report if there's an actual disagreement on non-hole cells
      if (hasDisagreement && !allHoles && Object.keys(detections).length >= 2) {
        const values: Record<string, string | null> = {};
        for (const [model, detection] of Object.entries(detections)) {
          values[model] = detection.region;
        }

        disagreements.push({
          id: generateDisagreementId('region_assignment', row, col),
          type: 'region_assignment',
          severity: 'warning',
          coordinate: { row, col },
          description: `Region disagreement at (${row},${col}): ${Object.entries(values).map(([m, v]) => `${m}=${v || 'null'}`).join(', ')}`,
          detections,
          values,
        });
      }
    }
  }

  return disagreements;
}

/**
 * Compare constraint definitions across models
 * Returns constraint disagreements for type, value, or operator mismatches
 */
function compareConstraints(
  modelResults: NormalizedModelResult[]
): ConstraintDisagreement[] {
  const disagreements: ConstraintDisagreement[] = [];
  const successfulResults = modelResults.filter(r => r.success && r.constraints);

  if (successfulResults.length < 2) return disagreements;

  // Collect all region labels across all models
  const allRegions = new Set<string>();
  for (const result of successfulResults) {
    if (result.constraints) {
      for (const region of Object.keys(result.constraints)) {
        allRegions.add(region);
      }
    }
  }

  // Compare constraints for each region
  for (const region of allRegions) {
    const constraintsByModel: Record<string, ConstraintDef | null> = {};

    for (const result of successfulResults) {
      constraintsByModel[result.model] = result.constraints?.[region] ?? null;
    }

    // Check for type disagreement
    const types = new Set<string | undefined>();
    for (const constraint of Object.values(constraintsByModel)) {
      types.add(constraint?.type);
    }

    if (types.size > 1) {
      disagreements.push({
        id: generateDisagreementId('constraint_type', region),
        type: 'constraint_type',
        severity: 'warning',
        region,
        description: `Constraint type disagreement for region ${region}: ${Object.entries(constraintsByModel).map(([m, c]) => `${m}=${c?.type || 'none'}`).join(', ')}`,
        values: constraintsByModel,
      });
    }

    // Check for value disagreement (when type is 'sum')
    const values = new Map<string, number | undefined>();
    for (const [model, constraint] of Object.entries(constraintsByModel)) {
      if (constraint?.type === 'sum') {
        values.set(model, constraint.value);
      }
    }

    if (values.size > 1) {
      const uniqueValues = new Set(values.values());
      if (uniqueValues.size > 1) {
        disagreements.push({
          id: generateDisagreementId('constraint_value', region),
          type: 'constraint_value',
          severity: 'warning',
          region,
          description: `Constraint value disagreement for region ${region}: ${Array.from(values.entries()).map(([m, v]) => `${m}=${v ?? 'undefined'}`).join(', ')}`,
          values: constraintsByModel,
        });
      }
    }

    // Check for operator disagreement
    const operators = new Map<string, string | undefined>();
    for (const [model, constraint] of Object.entries(constraintsByModel)) {
      if (constraint?.type === 'sum') {
        operators.set(model, constraint.op);
      }
    }

    if (operators.size > 1) {
      const uniqueOps = new Set(operators.values());
      if (uniqueOps.size > 1) {
        disagreements.push({
          id: generateDisagreementId('constraint_operator', region),
          type: 'constraint_operator',
          severity: 'warning',
          region,
          description: `Constraint operator disagreement for region ${region}: ${Array.from(operators.entries()).map(([m, o]) => `${m}=${o || '=='}`).join(', ')}`,
          values: constraintsByModel,
        });
      }
    }
  }

  return disagreements;
}

/**
 * Compare domino lists across models (order-independent)
 * Returns domino disagreements for count mismatches and individual differences
 */
function compareDominoes(
  boardResults: NormalizedModelResult[],
  dominoResponses: DominoModelResponse[]
): DominoDisagreement[] {
  const disagreements: DominoDisagreement[] = [];
  const successfulResponses = dominoResponses.filter(r => r.parseSuccess && r.parsedData);

  if (successfulResponses.length < 2) return disagreements;

  // Check for count disagreement
  const counts: Record<string, number> = {};
  let hasCountDisagreement = false;
  let firstCount: number | undefined;

  for (const response of successfulResponses) {
    const count = response.parsedData!.dominoes.length;
    counts[response.model] = count;

    if (firstCount === undefined) {
      firstCount = count;
    } else if (count !== firstCount) {
      hasCountDisagreement = true;
    }
  }

  if (hasCountDisagreement) {
    disagreements.push({
      id: generateDisagreementId('domino_count'),
      type: 'domino_count',
      severity: 'critical',
      description: `Domino count disagreement: ${Object.entries(counts).map(([m, c]) => `${m}=${c}`).join(', ')}`,
      values: counts,
    });
  }

  // Compare individual dominoes (order-independent)
  // Build normalized set for each model
  const dominoSetsByModel: Record<string, Set<string>> = {};
  for (const response of successfulResponses) {
    const normalized = new Set(response.parsedData!.dominoes.map(normalizeDomino));
    dominoSetsByModel[response.model] = normalized;
  }

  // Find dominoes present in some models but not others
  const allDominoes = new Set<string>();
  for (const set of Object.values(dominoSetsByModel)) {
    for (const d of set) {
      allDominoes.add(d);
    }
  }

  for (const domino of allDominoes) {
    const modelsWithDomino: string[] = [];
    const modelsWithoutDomino: string[] = [];

    for (const [model, set] of Object.entries(dominoSetsByModel)) {
      if (set.has(domino)) {
        modelsWithDomino.push(model);
      } else {
        modelsWithoutDomino.push(model);
      }
    }

    if (modelsWithDomino.length > 0 && modelsWithoutDomino.length > 0) {
      const [pip1, pip2] = domino.split('-').map(Number) as [number, number];
      const values: Record<string, boolean> = {};
      for (const m of modelsWithDomino) values[m] = true;
      for (const m of modelsWithoutDomino) values[m] = false;

      disagreements.push({
        id: generateDisagreementId('domino_value', pip1, pip2),
        type: 'domino_value',
        severity: 'warning',
        description: `Domino [${pip1},${pip2}] disagreement: present in ${modelsWithDomino.join(', ')} but missing in ${modelsWithoutDomino.join(', ')}`,
        values,
      });
    }
  }

  return disagreements;
}

/**
 * Build cell disagreement map for quick lookups
 */
function buildCellDisagreementMap(
  holeDisagreements: CellDisagreement[],
  regionDisagreements: CellDisagreement[]
): Map<string, CellDisagreement[]> {
  const map = new Map<string, CellDisagreement[]>();

  for (const d of [...holeDisagreements, ...regionDisagreements]) {
    const key = cellKey(d.coordinate.row, d.coordinate.col);
    const existing = map.get(key) || [];
    existing.push(d);
    map.set(key, existing);
  }

  return map;
}

/**
 * Calculate summary statistics from all disagreements
 */
function calculateSummary(allDisagreements: Disagreement[]): DisagreementSummary {
  const summary: DisagreementSummary = {
    total: allDisagreements.length,
    critical: 0,
    warning: 0,
    info: 0,
  };

  for (const d of allDisagreements) {
    summary[d.severity]++;
  }

  return summary;
}

// ════════════════════════════════════════════════════════════════════════════
// Main Comparison Function
// ════════════════════════════════════════════════════════════════════════════

/**
 * Compare multiple model extraction results and identify disagreements
 *
 * This is the core comparison logic for the extraction comparison/diff view.
 * It analyzes board and domino model responses to find:
 * - Grid dimension mismatches (critical)
 * - Hole position disagreements (critical)
 * - Region assignment differences (warning)
 * - Constraint type/value/operator differences (warning)
 * - Domino count and individual value differences (critical/warning)
 *
 * @param boardResponses - Per-model board extraction responses
 * @param dominoResponses - Per-model domino extraction responses
 * @param options - Optional comparison options
 * @returns Comprehensive comparison result with all disagreements
 */
export function compareCellDetections(
  boardResponses: BoardComparisonInput,
  dominoResponses: DominoComparisonInput,
  options?: ComparisonOptions
): ComparisonResult {
  const modelsCompared: string[] = [];

  // Track all models that participated
  for (const r of boardResponses) {
    if (!modelsCompared.includes(r.model)) {
      modelsCompared.push(r.model);
    }
  }
  for (const r of dominoResponses) {
    if (!modelsCompared.includes(r.model)) {
      modelsCompared.push(r.model);
    }
  }

  // Handle edge cases
  if (modelsCompared.length < 2) {
    return createEmptyComparisonResult(modelsCompared);
  }

  // Normalize board responses
  const modelResults: NormalizedModelResult[] = boardResponses.map(r =>
    normalizeBoardResponse(r as BoardModelResponse)
  );

  // Determine max dimensions for comparison grid
  let maxRows = 0;
  let maxCols = 0;
  for (const result of modelResults) {
    if (result.success && result.dimensions) {
      maxRows = Math.max(maxRows, result.dimensions.rows);
      maxCols = Math.max(maxCols, result.dimensions.cols);
    }
  }

  // Compare different aspects
  const dimensionDisagreements = compareDimensions(modelResults);
  const holeDisagreements = compareHolePositions(modelResults, maxRows, maxCols);
  const regionDisagreements = compareRegionAssignments(modelResults, maxRows, maxCols);
  const constraintDisagreements = compareConstraints(modelResults);
  const dominoDisagreements = compareDominoes(modelResults, dominoResponses as DominoModelResponse[]);

  // Add domino info to model results
  for (const result of modelResults) {
    const dominoResponse = dominoResponses.find(r => r.model === result.model) as DominoModelResponse | undefined;
    if (dominoResponse?.parseSuccess && dominoResponse.parsedData) {
      result.dominoes = dominoResponse.parsedData.dominoes;
      result.confidence = {
        ...result.confidence,
        dominoes: dominoResponse.parsedData.confidence,
      };
    }
  }

  // Build disagreement collections
  const disagreementsByType: DisagreementsByType = {
    gridDimensions: dimensionDisagreements,
    holePositions: holeDisagreements,
    regionAssignments: regionDisagreements,
    constraints: constraintDisagreements,
    dominoes: dominoDisagreements,
  };

  const allDisagreements: Disagreement[] = [
    ...dimensionDisagreements,
    ...holeDisagreements,
    ...regionDisagreements,
    ...constraintDisagreements,
    ...dominoDisagreements,
  ];

  // Filter by severity if option set
  const filteredDisagreements = options?.includeInfoLevel === false
    ? allDisagreements.filter(d => d.severity !== 'info')
    : allDisagreements;

  // Build quick lookup map
  const cellDisagreementMap = buildCellDisagreementMap(holeDisagreements, regionDisagreements);

  // Calculate summary
  const summary = calculateSummary(filteredDisagreements);

  return {
    summary,
    disagreementsByType,
    allDisagreements: filteredDisagreements,
    cellDisagreementMap,
    modelResults,
    modelsCompared,
    selectedModel: options?.referenceModel,
    isUnanimous: filteredDisagreements.length === 0,
    comparedAt: Date.now(),
  };
}

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

  /** Per-model domino values */
  values: Record<string, DominoPair[] | DominoPair | number>;
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

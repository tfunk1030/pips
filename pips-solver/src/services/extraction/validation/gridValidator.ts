/**
 * Grid Validator
 *
 * Validates grid geometry and cell detection results
 * against NYT Pips-specific rules with cross-validation
 * between extraction stages.
 */

import {
  GridGeometryResult,
  CellDetectionResult,
  ValidationResult,
  ValidationError,
} from '../types';
import { NYT_VALIDATION } from '../config';

// =============================================================================
// Configuration Constants
// =============================================================================

/** Minimum ratio of cells to grid area for a valid puzzle */
const MIN_CELL_DENSITY = 0.3;

/** Maximum ratio of holes (too many holes suggests detection error) */
const MAX_HOLE_DENSITY = 0.7;

/** Confidence difference threshold for flagging inconsistency */
const CONFIDENCE_INCONSISTENCY_THRESHOLD = 0.3;

/** Low confidence warning threshold */
const LOW_CONFIDENCE_THRESHOLD = 0.6;

// =============================================================================
// Basic Validation Functions
// =============================================================================

/**
 * Validate grid geometry result
 */
export function validateGridGeometry(result: GridGeometryResult): ValidationResult {
  const errors: ValidationError[] = [];
  const { minGridSize, maxGridSize } = NYT_VALIDATION;

  // Check rows in range
  if (result.rows < minGridSize) {
    errors.push({
      stage: 'grid',
      field: 'rows',
      message: `Rows (${result.rows}) below minimum (${minGridSize})`,
      value: result.rows,
    });
  }

  if (result.rows > maxGridSize) {
    errors.push({
      stage: 'grid',
      field: 'rows',
      message: `Rows (${result.rows}) above maximum (${maxGridSize})`,
      value: result.rows,
    });
  }

  // Check cols in range
  if (result.cols < minGridSize) {
    errors.push({
      stage: 'grid',
      field: 'cols',
      message: `Cols (${result.cols}) below minimum (${minGridSize})`,
      value: result.cols,
    });
  }

  if (result.cols > maxGridSize) {
    errors.push({
      stage: 'grid',
      field: 'cols',
      message: `Cols (${result.cols}) above maximum (${maxGridSize})`,
      value: result.cols,
    });
  }

  // Check confidence is valid
  if (result.confidence < 0 || result.confidence > 1) {
    errors.push({
      stage: 'grid',
      field: 'confidence',
      message: `Confidence (${result.confidence}) must be between 0 and 1`,
      value: result.confidence,
    });
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validate cell detection result
 */
export function validateCellDetection(
  result: CellDetectionResult,
  grid: GridGeometryResult
): ValidationResult {
  const errors: ValidationError[] = [];

  // Parse shape
  const lines = result.shape.split('\n').filter((l) => l.length > 0);

  // Check row count matches
  if (lines.length !== grid.rows) {
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `Shape has ${lines.length} rows, expected ${grid.rows}`,
      value: lines.length,
    });
  }

  // Check each row
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Check column count
    if (line.length !== grid.cols) {
      errors.push({
        stage: 'cells',
        field: 'shape',
        message: `Row ${i + 1} has ${line.length} cols, expected ${grid.cols}`,
        value: line.length,
      });
    }

    // Check valid characters
    if (!/^[.#]+$/.test(line)) {
      const invalidChars = line.replace(/[.#]/g, '');
      errors.push({
        stage: 'cells',
        field: 'shape',
        message: `Row ${i + 1} contains invalid characters: "${invalidChars}"`,
        value: invalidChars,
      });
    }
  }

  // Count cells
  const cellCount = (result.shape.match(/\./g) || []).length;

  // Cells must be even (for domino pairs)
  if (cellCount % 2 !== 0) {
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `Cell count (${cellCount}) must be even for domino pairs`,
      value: cellCount,
    });
  }

  // Minimum cells
  if (cellCount < NYT_VALIDATION.minCells) {
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `Cell count (${cellCount}) below minimum (${NYT_VALIDATION.minCells})`,
      value: cellCount,
    });
  }

  // Maximum cells (grid area)
  const maxCells = grid.rows * grid.cols;
  if (cellCount > maxCells) {
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `Cell count (${cellCount}) exceeds grid area (${maxCells})`,
      value: cellCount,
    });
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

// =============================================================================
// Cross-Validation Functions
// =============================================================================

/**
 * Cross-validate grid geometry and cell detection results
 *
 * Performs comprehensive checks to ensure consistency between stages:
 * - Cell connectivity (cells form connected components)
 * - Hole pattern analysis (suspicious patterns like full rows/cols)
 * - Grid utilization (reasonable cell density)
 * - Confidence consistency between stages
 */
export function crossValidateGridAndCells(
  grid: GridGeometryResult,
  cells: CellDetectionResult
): ValidationResult {
  const errors: ValidationError[] = [];

  // Parse shape into 2D array
  const shapeLines = cells.shape.split('\n').filter((l) => l.length > 0);

  // Basic dimension check (prerequisite for other checks)
  if (shapeLines.length !== grid.rows) {
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `Cross-validation failed: shape rows (${shapeLines.length}) don't match grid rows (${grid.rows})`,
      value: { shapeRows: shapeLines.length, gridRows: grid.rows },
    });
    // Can't continue cross-validation with mismatched dimensions
    return { valid: false, errors };
  }

  for (let i = 0; i < shapeLines.length; i++) {
    if (shapeLines[i].length !== grid.cols) {
      errors.push({
        stage: 'cells',
        field: 'shape',
        message: `Cross-validation failed: row ${i + 1} cols (${shapeLines[i].length}) don't match grid cols (${grid.cols})`,
        value: { rowCols: shapeLines[i].length, gridCols: grid.cols },
      });
    }
  }

  // If dimensions are wrong, stop here
  if (errors.length > 0) {
    return { valid: false, errors };
  }

  // Check cell connectivity
  const connectivityErrors = checkCellConnectivity(shapeLines);
  errors.push(...connectivityErrors);

  // Check for suspicious hole patterns
  const holePatternErrors = checkHolePatterns(shapeLines, grid);
  errors.push(...holePatternErrors);

  // Check grid area utilization
  const utilizationErrors = checkGridUtilization(shapeLines, grid);
  errors.push(...utilizationErrors);

  // Check confidence consistency
  const confidenceErrors = checkConfidenceConsistency(grid, cells);
  errors.push(...confidenceErrors);

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Check that all cells form connected components
 *
 * Uses BFS to verify cells are reachable from each other.
 * A valid puzzle should have cells that can be covered by dominoes,
 * meaning isolated single cells are invalid.
 */
function checkCellConnectivity(shapeLines: string[]): ValidationError[] {
  const errors: ValidationError[] = [];
  const rows = shapeLines.length;
  if (rows === 0) return errors;

  const cols = shapeLines[0].length;

  // Find all cell positions
  const cellPositions: [number, number][] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (shapeLines[r][c] === '.') {
        cellPositions.push([r, c]);
      }
    }
  }

  if (cellPositions.length === 0) {
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: 'No cells found in shape - entire grid is holes',
      value: 0,
    });
    return errors;
  }

  // BFS from first cell to find all connected cells
  const visited = new Set<string>();
  const queue: [number, number][] = [cellPositions[0]];
  visited.add(`${cellPositions[0][0]},${cellPositions[0][1]}`);

  while (queue.length > 0) {
    const [r, c] = queue.shift()!;

    // Check 4-connected neighbors
    const neighbors: [number, number][] = [
      [r - 1, c],
      [r + 1, c],
      [r, c - 1],
      [r, c + 1],
    ];

    for (const [nr, nc] of neighbors) {
      const key = `${nr},${nc}`;
      if (visited.has(key)) continue;

      // Check bounds and if it's a cell
      if (
        nr >= 0 &&
        nr < rows &&
        nc >= 0 &&
        nc < cols &&
        shapeLines[nr][nc] === '.'
      ) {
        visited.add(key);
        queue.push([nr, nc]);
      }
    }
  }

  // Check for disconnected cells
  if (visited.size !== cellPositions.length) {
    const disconnectedCount = cellPositions.length - visited.size;
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `Found ${disconnectedCount} disconnected cell(s) - puzzle cells should form connected regions`,
      value: { connected: visited.size, total: cellPositions.length },
    });

    // Find isolated cells for detailed feedback
    const isolatedCells: [number, number][] = [];
    for (const [r, c] of cellPositions) {
      if (!visited.has(`${r},${c}`)) {
        isolatedCells.push([r, c]);
      }
    }

    if (isolatedCells.length <= 5) {
      const positions = isolatedCells
        .map(([r, c]) => `(${r + 1},${c + 1})`)
        .join(', ');
      errors.push({
        stage: 'cells',
        field: 'shape',
        message: `Isolated cells at: ${positions}`,
        value: isolatedCells,
      });
    }
  }

  return errors;
}

/**
 * Check for suspicious hole patterns that suggest detection errors
 *
 * Detects:
 * - Full rows of holes (except at edges which could be L-shapes)
 * - Full columns of holes (except at edges)
 * - Symmetric hole patterns that look artificial
 */
function checkHolePatterns(
  shapeLines: string[],
  grid: GridGeometryResult
): ValidationError[] {
  const errors: ValidationError[] = [];
  const rows = grid.rows;
  const cols = grid.cols;

  // Check for full rows of holes (suspicious if in middle)
  const fullHoleRows: number[] = [];
  for (let r = 0; r < rows; r++) {
    if (shapeLines[r].split('').every((c) => c === '#')) {
      fullHoleRows.push(r);
    }
  }

  // Full hole rows in the middle are suspicious (not at edges)
  const middleHoleRows = fullHoleRows.filter((r) => r > 0 && r < rows - 1);
  if (middleHoleRows.length > 0) {
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `Suspicious: full row(s) of holes in middle at row(s) ${middleHoleRows.map((r) => r + 1).join(', ')}`,
      value: middleHoleRows,
    });
  }

  // Check for full columns of holes (suspicious if in middle)
  const fullHoleCols: number[] = [];
  for (let c = 0; c < cols; c++) {
    let allHoles = true;
    for (let r = 0; r < rows; r++) {
      if (shapeLines[r][c] !== '#') {
        allHoles = false;
        break;
      }
    }
    if (allHoles) {
      fullHoleCols.push(c);
    }
  }

  // Full hole columns in the middle are suspicious
  const middleHoleCols = fullHoleCols.filter((c) => c > 0 && c < cols - 1);
  if (middleHoleCols.length > 0) {
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `Suspicious: full column(s) of holes in middle at col(s) ${middleHoleCols.map((c) => c + 1).join(', ')}`,
      value: middleHoleCols,
    });
  }

  // Check for checkerboard pattern (alternating holes - definitely an error)
  let checkerCount = 0;
  let totalChecked = 0;
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const tl = shapeLines[r][c];
      const tr = shapeLines[r][c + 1];
      const bl = shapeLines[r + 1][c];
      const br = shapeLines[r + 1][c + 1];

      // Check for 2x2 checkerboard pattern
      if (
        (tl === '.' && tr === '#' && bl === '#' && br === '.') ||
        (tl === '#' && tr === '.' && bl === '.' && br === '#')
      ) {
        checkerCount++;
      }
      totalChecked++;
    }
  }

  // More than 30% checkerboard 2x2 patterns is suspicious
  if (totalChecked > 0 && checkerCount / totalChecked > 0.3) {
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `Suspicious checkerboard pattern detected (${Math.round((checkerCount / totalChecked) * 100)}% of 2x2 regions)`,
      value: { checkerCount, totalChecked },
    });
  }

  return errors;
}

/**
 * Check grid area utilization
 *
 * Validates that the ratio of cells to total area is reasonable:
 * - Too few cells might indicate over-detection of holes
 * - All cells (no holes) might indicate under-detection of holes
 */
function checkGridUtilization(
  shapeLines: string[],
  grid: GridGeometryResult
): ValidationError[] {
  const errors: ValidationError[] = [];

  const totalArea = grid.rows * grid.cols;
  let cellCount = 0;
  let holeCount = 0;

  for (const line of shapeLines) {
    for (const char of line) {
      if (char === '.') cellCount++;
      else if (char === '#') holeCount++;
    }
  }

  const cellDensity = cellCount / totalArea;
  const holeDensity = holeCount / totalArea;

  // Check for too few cells
  if (cellDensity < MIN_CELL_DENSITY) {
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `Low cell density (${Math.round(cellDensity * 100)}%) - possible over-detection of holes`,
      value: { cellDensity, cellCount, totalArea },
    });
  }

  // Check for too many holes
  if (holeDensity > MAX_HOLE_DENSITY) {
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `High hole density (${Math.round(holeDensity * 100)}%) - verify hole detection accuracy`,
      value: { holeDensity, holeCount, totalArea },
    });
  }

  // Check for no holes at all (unusual for NYT puzzles)
  if (holeCount === 0 && totalArea > 16) {
    // No holes in larger grids is unusual
    errors.push({
      stage: 'cells',
      field: 'shape',
      message: `No holes detected in ${grid.rows}x${grid.cols} grid - most NYT puzzles have irregular shapes`,
      value: { holeCount: 0, totalArea },
    });
  }

  return errors;
}

/**
 * Check confidence consistency between grid and cell detection stages
 *
 * Large confidence differences between stages may indicate:
 * - One stage had difficulty with the image
 * - Possible extraction errors that need review
 */
function checkConfidenceConsistency(
  grid: GridGeometryResult,
  cells: CellDetectionResult
): ValidationError[] {
  const errors: ValidationError[] = [];

  const confidenceDiff = Math.abs(grid.confidence - cells.confidence);

  // Flag large confidence differences
  if (confidenceDiff > CONFIDENCE_INCONSISTENCY_THRESHOLD) {
    const lowerStage = grid.confidence < cells.confidence ? 'grid' : 'cells';
    errors.push({
      stage: lowerStage as 'grid' | 'cells',
      field: 'confidence',
      message: `Large confidence gap between stages: grid=${(grid.confidence * 100).toFixed(0)}%, cells=${(cells.confidence * 100).toFixed(0)}%`,
      value: { gridConfidence: grid.confidence, cellsConfidence: cells.confidence },
    });
  }

  // Warn on low confidence in either stage
  if (grid.confidence < LOW_CONFIDENCE_THRESHOLD) {
    errors.push({
      stage: 'grid',
      field: 'confidence',
      message: `Low grid detection confidence (${(grid.confidence * 100).toFixed(0)}%) - manual verification recommended`,
      value: grid.confidence,
    });
  }

  if (cells.confidence < LOW_CONFIDENCE_THRESHOLD) {
    errors.push({
      stage: 'cells',
      field: 'confidence',
      message: `Low cell detection confidence (${(cells.confidence * 100).toFixed(0)}%) - manual verification recommended`,
      value: cells.confidence,
    });
  }

  return errors;
}

// =============================================================================
// Multi-Model Validation Functions
// =============================================================================

/**
 * Check if grid dimensions suggest rotation issue
 * (e.g., 2 models say 6x5, 1 says 5x6)
 */
export function checkRotationMismatch(
  results: GridGeometryResult[]
): { hasRotation: boolean; suggestion: string } {
  if (results.length < 2) {
    return { hasRotation: false, suggestion: '' };
  }

  const dims = results.map((r) => ({ rows: r.rows, cols: r.cols }));

  // Check for swapped dimensions
  for (let i = 0; i < dims.length; i++) {
    for (let j = i + 1; j < dims.length; j++) {
      if (dims[i].rows === dims[j].cols && dims[i].cols === dims[j].rows) {
        return {
          hasRotation: true,
          suggestion: `Possible rotation: ${dims[i].rows}x${dims[i].cols} vs ${dims[j].rows}x${dims[j].cols}. Check image orientation.`,
        };
      }
    }
  }

  return { hasRotation: false, suggestion: '' };
}

/**
 * Compare cell detection results from multiple models
 *
 * Identifies disagreements and returns detailed diff information
 * for user review.
 */
export function compareCellDetections(
  results: CellDetectionResult[]
): {
  agreement: number;
  disagreements: { row: number; col: number; values: string[] }[];
} {
  if (results.length < 2) {
    return { agreement: 1.0, disagreements: [] };
  }

  // Parse all shapes
  const shapes = results.map((r) =>
    r.shape.split('\n').filter((l) => l.length > 0)
  );

  // Find dimensions (use first result)
  const rows = shapes[0].length;
  const cols = shapes[0][0]?.length || 0;

  const disagreements: { row: number; col: number; values: string[] }[] = [];
  let agreements = 0;
  let total = 0;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const values = shapes.map((s) => (s[r] && s[r][c]) || '?');
      total++;

      // Check if all values agree
      const uniqueValues = Array.from(new Set(values));
      if (uniqueValues.length === 1) {
        agreements++;
      } else {
        disagreements.push({ row: r + 1, col: c + 1, values });
      }
    }
  }

  return {
    agreement: total > 0 ? agreements / total : 1.0,
    disagreements,
  };
}

/**
 * Validate consistency across multiple grid results
 *
 * Checks for dimension agreement and flags outliers
 */
export function validateGridConsensus(
  results: GridGeometryResult[]
): ValidationResult {
  const errors: ValidationError[] = [];

  if (results.length < 2) {
    return { valid: true, errors: [] };
  }

  // Group by dimensions
  const dimGroups = new Map<string, number>();
  for (const result of results) {
    const key = `${result.rows}x${result.cols}`;
    dimGroups.set(key, (dimGroups.get(key) || 0) + 1);
  }

  // Find majority
  let majorityDim = '';
  let majorityCount = 0;
  const dimEntries = Array.from(dimGroups.entries());
  for (const [dim, count] of dimEntries) {
    if (count > majorityCount) {
      majorityCount = count;
      majorityDim = dim;
    }
  }

  // Check for disagreement
  if (dimGroups.size > 1) {
    const dimList = Array.from(dimGroups.entries())
      .map(([dim, count]) => `${dim} (${count}x)`)
      .join(', ');

    errors.push({
      stage: 'grid',
      field: 'dimensions',
      message: `Model disagreement on grid dimensions: ${dimList}`,
      value: Object.fromEntries(dimGroups),
    });
  }

  // Check for rotation issue
  const rotationCheck = checkRotationMismatch(results);
  if (rotationCheck.hasRotation) {
    errors.push({
      stage: 'grid',
      field: 'orientation',
      message: rotationCheck.suggestion,
      value: results.map((r) => ({ rows: r.rows, cols: r.cols })),
    });
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

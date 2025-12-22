/**
 * Grid Validator
 *
 * Validates grid geometry and cell detection results
 * against NYT Pips-specific rules.
 */

import { GridGeometryResult, CellDetectionResult, ValidationResult, ValidationError } from '../types';
import { NYT_VALIDATION } from '../config';

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

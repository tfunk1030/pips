/**
 * Region Validator
 *
 * Validates region mapping and constraint extraction results.
 */

import {
  RegionMappingResult,
  CellDetectionResult,
  GridGeometryResult,
  ConstraintExtractionResult,
  Constraint,
  ValidationResult,
  ValidationError,
} from '../types';
import { NYT_VALIDATION } from '../config';

/**
 * Validate region mapping result
 */
export function validateRegionMapping(
  result: RegionMappingResult,
  grid: GridGeometryResult,
  cells: CellDetectionResult
): ValidationResult {
  const errors: ValidationError[] = [];

  const regionLines = result.regions.split('\n').filter((l) => l.length > 0);
  const shapeLines = cells.shape.split('\n').filter((l) => l.length > 0);

  // Check row count matches
  if (regionLines.length !== grid.rows) {
    errors.push({
      stage: 'regions',
      field: 'regions',
      message: `Regions has ${regionLines.length} rows, expected ${grid.rows}`,
      value: regionLines.length,
    });
  }

  // Track region sizes
  const regionSizes = new Map<string, number>();

  // Check each position
  for (let i = 0; i < Math.max(regionLines.length, shapeLines.length); i++) {
    const regionLine = regionLines[i] || '';
    const shapeLine = shapeLines[i] || '';

    // Check column count
    if (regionLine.length !== grid.cols) {
      errors.push({
        stage: 'regions',
        field: 'regions',
        message: `Row ${i + 1} has ${regionLine.length} cols, expected ${grid.cols}`,
        value: regionLine.length,
      });
      continue;
    }

    // Check character by character
    for (let j = 0; j < grid.cols; j++) {
      const regionChar = regionLine[j];
      const shapeChar = shapeLine[j];

      // Holes must match
      if (shapeChar === '#') {
        if (regionChar !== '#') {
          errors.push({
            stage: 'regions',
            field: 'regions',
            message: `Position (${i + 1},${j + 1}) is hole in shape but '${regionChar}' in regions`,
            value: regionChar,
          });
        }
      } else {
        // Cells must be A-Z
        if (!/[A-Z]/.test(regionChar)) {
          errors.push({
            stage: 'regions',
            field: 'regions',
            message: `Position (${i + 1},${j + 1}) should be A-Z but is '${regionChar}'`,
            value: regionChar,
          });
        } else {
          // Count region size
          regionSizes.set(regionChar, (regionSizes.get(regionChar) || 0) + 1);
        }
      }
    }
  }

  // Check each region has at least 2 cells
  for (const [region, size] of regionSizes) {
    if (size < 2) {
      errors.push({
        stage: 'regions',
        field: 'regions',
        message: `Region ${region} has only ${size} cell(s) - needs at least 2 for a domino`,
        value: size,
      });
    }
  }

  // Check contiguity (each region should be connected)
  const contiguityErrors = checkRegionContiguity(regionLines, regionSizes);
  errors.push(...contiguityErrors);

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Check that each region is contiguous (all cells connected)
 */
function checkRegionContiguity(
  regionLines: string[],
  regionSizes: Map<string, number>
): ValidationError[] {
  const errors: ValidationError[] = [];

  for (const region of regionSizes.keys()) {
    // Find all cells for this region
    const cells: [number, number][] = [];

    for (let i = 0; i < regionLines.length; i++) {
      for (let j = 0; j < regionLines[i].length; j++) {
        if (regionLines[i][j] === region) {
          cells.push([i, j]);
        }
      }
    }

    if (cells.length === 0) continue;

    // BFS to check connectivity
    const visited = new Set<string>();
    const queue: [number, number][] = [cells[0]];
    visited.add(`${cells[0][0]},${cells[0][1]}`);

    while (queue.length > 0) {
      const [r, c] = queue.shift()!;

      // Check 4 neighbors
      const neighbors: [number, number][] = [
        [r - 1, c],
        [r + 1, c],
        [r, c - 1],
        [r, c + 1],
      ];

      for (const [nr, nc] of neighbors) {
        const key = `${nr},${nc}`;
        if (visited.has(key)) continue;

        // Check if this neighbor is in our region
        if (
          nr >= 0 &&
          nr < regionLines.length &&
          nc >= 0 &&
          nc < regionLines[nr].length &&
          regionLines[nr][nc] === region
        ) {
          visited.add(key);
          queue.push([nr, nc]);
        }
      }
    }

    // All cells should be visited
    if (visited.size !== cells.length) {
      errors.push({
        stage: 'regions',
        field: 'regions',
        message: `Region ${region} is not contiguous - found ${visited.size} connected cells but ${cells.length} total`,
        value: { connected: visited.size, total: cells.length },
      });
    }
  }

  return errors;
}

/**
 * Validate constraint extraction result
 */
export function validateConstraints(
  result: ConstraintExtractionResult,
  regions: RegionMappingResult
): ValidationResult {
  const errors: ValidationError[] = [];

  // Get valid region labels
  const validLabels = new Set<string>();
  for (const char of regions.regions) {
    if (char !== '#' && char !== '\n') {
      validLabels.add(char);
    }
  }

  // Count cells per region for sum validation
  const regionSizes = new Map<string, number>();
  for (const char of regions.regions) {
    if (char !== '#' && char !== '\n') {
      regionSizes.set(char, (regionSizes.get(char) || 0) + 1);
    }
  }

  for (const [region, constraint] of Object.entries(result.constraints)) {
    // Check region exists
    if (!validLabels.has(region)) {
      errors.push({
        stage: 'constraints',
        field: 'constraints',
        message: `Constraint for unknown region "${region}"`,
        value: region,
      });
      continue;
    }

    // Validate based on type
    if (constraint.type === 'sum') {
      // Check operator
      if (!constraint.op || !['==', '<', '>'].includes(constraint.op)) {
        errors.push({
          stage: 'constraints',
          field: 'constraints',
          message: `Region ${region}: invalid operator "${constraint.op}"`,
          value: constraint.op,
        });
      }

      // Check value range
      if (constraint.value === undefined) {
        errors.push({
          stage: 'constraints',
          field: 'constraints',
          message: `Region ${region}: sum constraint missing value`,
          value: undefined,
        });
      } else {
        if (constraint.value < 0) {
          errors.push({
            stage: 'constraints',
            field: 'constraints',
            message: `Region ${region}: sum value ${constraint.value} cannot be negative`,
            value: constraint.value,
          });
        }

        if (constraint.value > NYT_VALIDATION.maxSumValue) {
          errors.push({
            stage: 'constraints',
            field: 'constraints',
            message: `Region ${region}: sum value ${constraint.value} exceeds maximum ${NYT_VALIDATION.maxSumValue}`,
            value: constraint.value,
          });
        }

        // Check if value is achievable given region size
        const regionSize = regionSizes.get(region) || 0;
        const maxPossible = regionSize * NYT_VALIDATION.pipRange[1];
        const minPossible = regionSize * NYT_VALIDATION.pipRange[0];

        if (constraint.op === '==' && constraint.value > maxPossible) {
          errors.push({
            stage: 'constraints',
            field: 'constraints',
            message: `Region ${region}: sum=${constraint.value} impossible (max ${maxPossible} for ${regionSize} cells)`,
            value: { target: constraint.value, max: maxPossible },
          });
        }

        if (constraint.op === '>' && constraint.value >= maxPossible) {
          errors.push({
            stage: 'constraints',
            field: 'constraints',
            message: `Region ${region}: sum>${constraint.value} impossible (max ${maxPossible})`,
            value: { target: constraint.value, max: maxPossible },
          });
        }
      }
    } else if (constraint.type === 'all_equal') {
      // all_equal should not have op or value
      if (constraint.op || constraint.value !== undefined) {
        // This is just a warning, not an error
        console.warn(
          `Region ${region}: all_equal constraint has unnecessary op/value`
        );
      }
    } else {
      errors.push({
        stage: 'constraints',
        field: 'constraints',
        message: `Region ${region}: unknown constraint type "${(constraint as Constraint).type}"`,
        value: (constraint as Constraint).type,
      });
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

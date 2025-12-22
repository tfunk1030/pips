/**
 * Validate puzzle specifications
 */

import { PuzzleSpec, ValidationResult } from '../model/types';

/**
 * Validate a puzzle specification for consistency
 */
export function validatePuzzleSpec(spec: PuzzleSpec): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Validate basic structure
  if (spec.rows <= 0) {
    errors.push('rows must be positive');
  }

  if (spec.cols <= 0) {
    errors.push('cols must be positive');
  }

  if (spec.maxPip !== undefined && spec.maxPip < 0) {
    errors.push('maxPip must be non-negative');
  }

  // Validate regions array
  if (!spec.regions || !Array.isArray(spec.regions)) {
    errors.push('regions must be a 2D array');
  } else {
    if (spec.regions.length !== spec.rows) {
      errors.push(`regions array must have ${spec.rows} rows, has ${spec.regions.length}`);
    }

    for (let r = 0; r < spec.regions.length; r++) {
      if (!Array.isArray(spec.regions[r])) {
        errors.push(`regions[${r}] must be an array`);
      } else if (spec.regions[r].length !== spec.cols) {
        errors.push(
          `regions[${r}] must have ${spec.cols} columns, has ${spec.regions[r].length}`
        );
      }
    }

    // Check that all region IDs are integers; allow -1 for holes
    const regionIds = new Set<number>();
    let holeCount = 0;
    for (let r = 0; r < spec.rows; r++) {
      for (let c = 0; c < spec.cols; c++) {
        const regionId = spec.regions[r]?.[c];
        if (typeof regionId !== 'number' || !Number.isInteger(regionId)) {
          errors.push(`Invalid region ID at (${r},${c}): ${regionId}`);
        } else if (regionId === -1) {
          holeCount++;
        } else {
          if (regionId < 0) {
            errors.push(`Invalid region ID at (${r},${c}): ${regionId}`);
            continue;
          }
          regionIds.add(regionId);
        }
      }
    }

    // Check that non-hole cells are covered
    const expectedCells = spec.rows * spec.cols - holeCount;
    let actualCells = 0;
    for (const regionId of regionIds) {
      let count = 0;
      for (let r = 0; r < spec.rows; r++) {
        for (let c = 0; c < spec.cols; c++) {
          if (spec.regions[r][c] === regionId) {
            count++;
          }
        }
      }
      actualCells += count;

      if (count === 0) {
        warnings.push(`Region ${regionId} is empty`);
      }
    }

    if (actualCells !== expectedCells) {
      errors.push(`Regions do not cover grid completely (${actualCells}/${expectedCells} cells)`);
    }
  }

  // Validate constraints
  if (spec.constraints) {
    for (const [regionIdStr, constraint] of Object.entries(spec.constraints)) {
      const regionId = parseInt(regionIdStr, 10);

      if (isNaN(regionId)) {
        errors.push(`Invalid region ID in constraints: ${regionIdStr}`);
        continue;
      }

      // Check if region exists
      let regionExists = false;
      for (let r = 0; r < spec.rows; r++) {
        for (let c = 0; c < spec.cols; c++) {
          if (spec.regions[r]?.[c] === regionId) {
            regionExists = true;
            break;
          }
        }
        if (regionExists) break;
      }

      if (!regionExists) {
        errors.push(`Constraint defined for non-existent region ${regionId}`);
      }

      // Validate constraint structure
      if (constraint.sum !== undefined) {
        if (typeof constraint.sum !== 'number' || constraint.sum < 0) {
          errors.push(`Invalid sum constraint for region ${regionId}: ${constraint.sum}`);
        }
      }

      if (constraint.op !== undefined) {
        if (!['=', '<', '>', '≠'].includes(constraint.op)) {
          errors.push(`Invalid operator for region ${regionId}: ${constraint.op}`);
        }

        if (constraint.value === undefined) {
          errors.push(`Operator constraint for region ${regionId} missing value`);
        } else if (typeof constraint.value !== 'number') {
          errors.push(`Invalid value for region ${regionId}: ${constraint.value}`);
        }
      }

      if (constraint.all_equal !== undefined) {
        if (typeof constraint.all_equal !== 'boolean') {
          errors.push(`Invalid all_equal for region ${regionId}: ${constraint.all_equal}`);
        }
      }

      if (constraint.all_different !== undefined) {
        if (typeof constraint.all_different !== 'boolean') {
          errors.push(`Invalid all_different for region ${regionId}: ${constraint.all_different}`);
        }
      }

      if (constraint.size !== undefined) {
        if (typeof constraint.size !== 'number' || constraint.size <= 0) {
          errors.push(`Invalid size constraint for region ${regionId}: ${constraint.size}`);
        } else {
          // Check actual region size
          let actualSize = 0;
          for (let r = 0; r < spec.rows; r++) {
            for (let c = 0; c < spec.cols; c++) {
              if (spec.regions[r]?.[c] === regionId) {
                actualSize++;
              }
            }
          }

          if (actualSize !== constraint.size) {
            warnings.push(
              `Region ${regionId} size mismatch: specified ${constraint.size}, actual ${actualSize}`
            );
          }
        }
      }

      // Check for conflicting constraints
      if (constraint.all_equal && constraint.all_different) {
        errors.push(`Region ${regionId} has conflicting all_equal and all_different constraints`);
      }
    }
  }

  // Check grid size is compatible with dominoes
  const totalCells = spec.regions
    ? spec.regions.flat().filter((rid) => rid !== -1).length
    : spec.rows * spec.cols;
  if (totalCells % 2 !== 0) {
    errors.push(
      `Grid has odd number of cells (${totalCells}). Dominoes require even number of cells.`
    );
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

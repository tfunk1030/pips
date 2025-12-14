/**
 * Validate puzzle solutions
 */

import {
  NormalizedPuzzle,
  Solution,
  ValidationResult,
  RegionCheckResult,
  DominoCheckResult,
  cellKey,
  isAdjacent,
} from '../model/types';
import { getRegionCells } from '../model/normalize';

/**
 * Validate a solution against the puzzle specification
 */
export function validateSolution(
  puzzle: NormalizedPuzzle,
  solution: Solution
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  const regionChecks: RegionCheckResult[] = [];
  const dominoChecks: DominoCheckResult[] = [];

  // Validate grid dimensions
  if (solution.gridPips.length !== puzzle.spec.rows) {
    errors.push(
      `Solution grid has ${solution.gridPips.length} rows, expected ${puzzle.spec.rows}`
    );
    return { valid: false, errors, warnings, regionChecks, dominoChecks };
  }

  for (let r = 0; r < solution.gridPips.length; r++) {
    if (solution.gridPips[r].length !== puzzle.spec.cols) {
      errors.push(
        `Solution grid row ${r} has ${solution.gridPips[r].length} cols, expected ${puzzle.spec.cols}`
      );
      return { valid: false, errors, warnings, regionChecks, dominoChecks };
    }
  }

  // Validate all cells are filled
  for (let r = 0; r < puzzle.spec.rows; r++) {
    for (let c = 0; c < puzzle.spec.cols; c++) {
      const value = solution.gridPips[r][c];
      if (value === null || value === undefined) {
        errors.push(`Cell (${r},${c}) is not filled`);
      } else {
        // Validate pip value is in valid range
        const maxPip = puzzle.spec.maxPip || 6;
        if (value < 0 || value > maxPip) {
          errors.push(`Cell (${r},${c}) has invalid pip value ${value} (max: ${maxPip})`);
        }
      }
    }
  }

  // Validate dominoes
  const cellToDomino = new Map<string, number>(); // cellKey -> domino index
  const dominoIds = new Set<string>();

  for (let i = 0; i < solution.dominoes.length; i++) {
    const placement = solution.dominoes[i];
    const check: DominoCheckResult = {
      valid: true,
      message: '',
      domino: placement,
    };

    // Check adjacency
    if (!isAdjacent(placement.cell1, placement.cell2)) {
      check.valid = false;
      check.message = `Domino ${i} cells are not adjacent`;
      errors.push(check.message);
    }

    // Check pip values match grid
    const gridPip1 = solution.gridPips[placement.cell1.row][placement.cell1.col];
    const gridPip2 = solution.gridPips[placement.cell2.row][placement.cell2.col];

    if (gridPip1 !== placement.domino.pip1) {
      check.valid = false;
      check.message = `Domino ${i} pip1 mismatch: domino=${placement.domino.pip1}, grid=${gridPip1}`;
      errors.push(check.message);
    }

    if (gridPip2 !== placement.domino.pip2) {
      check.valid = false;
      check.message = `Domino ${i} pip2 mismatch: domino=${placement.domino.pip2}, grid=${gridPip2}`;
      errors.push(check.message);
    }

    // Check domino ID is correct
    const expectedId =
      placement.domino.pip1 <= placement.domino.pip2
        ? `${placement.domino.pip1}-${placement.domino.pip2}`
        : `${placement.domino.pip2}-${placement.domino.pip1}`;

    if (placement.domino.id !== expectedId) {
      check.valid = false;
      check.message = `Domino ${i} has incorrect ID: ${placement.domino.id}, expected ${expectedId}`;
      errors.push(check.message);
    }

    // Check for duplicate domino usage (if not allowed)
    if (!puzzle.spec.allowDuplicates) {
      if (dominoIds.has(placement.domino.id)) {
        check.valid = false;
        check.message = `Domino ${placement.domino.id} used multiple times`;
        errors.push(check.message);
      }
      dominoIds.add(placement.domino.id);
    }

    // Check cells aren't used in multiple dominoes
    const key1 = cellKey(placement.cell1);
    const key2 = cellKey(placement.cell2);

    if (cellToDomino.has(key1)) {
      check.valid = false;
      check.message = `Cell ${key1} used in multiple dominoes`;
      errors.push(check.message);
    }

    if (cellToDomino.has(key2)) {
      check.valid = false;
      check.message = `Cell ${key2} used in multiple dominoes`;
      errors.push(check.message);
    }

    cellToDomino.set(key1, i);
    cellToDomino.set(key2, i);

    dominoChecks.push(check);
  }

  // Check all cells are covered by dominoes
  for (let r = 0; r < puzzle.spec.rows; r++) {
    for (let c = 0; c < puzzle.spec.cols; c++) {
      const key = cellKey({ row: r, col: c });
      if (!cellToDomino.has(key)) {
        errors.push(`Cell (${r},${c}) is not covered by any domino`);
      }
    }
  }

  // Validate region constraints
  for (const [regionIdStr, constraint] of Object.entries(puzzle.spec.constraints)) {
    const regionId = parseInt(regionIdStr, 10);
    const cells = getRegionCells(puzzle, regionId);
    const values = cells.map((c) => solution.gridPips[c.row][c.col]);

    const check: RegionCheckResult = {
      regionId,
      valid: true,
      constraint,
      actualValues: values,
      message: '',
    };

    // Check sum constraint
    if (constraint.sum !== undefined) {
      const actualSum = values.reduce((a, b) => a + b, 0);
      if (actualSum !== constraint.sum) {
        check.valid = false;
        check.message = `Region ${regionId} sum is ${actualSum}, expected ${constraint.sum}`;
        errors.push(check.message);
      } else {
        check.message = `Region ${regionId} sum=${actualSum} ✓`;
      }
    }

    // Check all_equal constraint
    if (constraint.all_equal) {
      const allEqual = values.every((v) => v === values[0]);
      if (!allEqual) {
        check.valid = false;
        check.message = `Region ${regionId} all_equal constraint violated: ${values.join(', ')}`;
        errors.push(check.message);
      } else {
        check.message = `Region ${regionId} all_equal=${values[0]} ✓`;
      }
    }

    // Check operator constraint
    if (constraint.op && constraint.value !== undefined) {
      const violators = values.filter((v) => {
        switch (constraint.op) {
          case '=':
            return v !== constraint.value;
          case '<':
            return v >= constraint.value!;
          case '>':
            return v <= constraint.value!;
          case '≠':
            return v === constraint.value;
          default:
            return false;
        }
      });

      if (violators.length > 0) {
        check.valid = false;
        check.message = `Region ${regionId} constraint "${constraint.op} ${constraint.value}" violated by: ${violators.join(', ')}`;
        errors.push(check.message);
      } else {
        check.message = `Region ${regionId} all ${constraint.op} ${constraint.value} ✓`;
      }
    }

    regionChecks.push(check);
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    regionChecks,
    dominoChecks,
  };
}

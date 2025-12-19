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

  // Validate all non-hole cells are filled (holes must be null)
  for (let r = 0; r < puzzle.spec.rows; r++) {
    for (let c = 0; c < puzzle.spec.cols; c++) {
      const isHole = puzzle.spec.regions[r]?.[c] === -1;
      const value = solution.gridPips[r][c];
      if (isHole) {
        if (value !== null) {
          errors.push(`Hole cell (${r},${c}) must be null`);
        }
        continue;
      }

      if (value === null || value === undefined) {
        errors.push(`Cell (${r},${c}) is not filled`);
        continue;
      }

      // Validate pip value is in valid range
      const maxPip = puzzle.spec.maxPip || 6;
      if (value < 0 || value > maxPip) {
        errors.push(`Cell (${r},${c}) has invalid pip value ${value} (max: ${maxPip})`);
      }
    }
  }

  // Validate dominoes
  const cellToDomino = new Map<string, number>(); // cellKey -> domino index

  // Tray inventory counts (if provided)
  const trayCounts = new Map<string, number>();
  if (Array.isArray(puzzle.spec.dominoes) && puzzle.spec.dominoes.length > 0) {
    for (const [a, b] of puzzle.spec.dominoes) {
      const id = a <= b ? `${a}-${b}` : `${b}-${a}`;
      trayCounts.set(id, (trayCounts.get(id) || 0) + 1);
    }
  }
  const usedCounts = new Map<string, number>();

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

    // Check cells are not holes
    const hole1 = puzzle.spec.regions[placement.cell1.row]?.[placement.cell1.col] === -1;
    const hole2 = puzzle.spec.regions[placement.cell2.row]?.[placement.cell2.col] === -1;
    if (hole1 || hole2) {
      check.valid = false;
      check.message = `Domino ${i} covers a hole cell`;
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
    usedCounts.set(placement.domino.id, (usedCounts.get(placement.domino.id) || 0) + 1);

    if (trayCounts.size > 0) {
      const available = trayCounts.get(placement.domino.id) || 0;
      const used = usedCounts.get(placement.domino.id) || 0;
      if (used > available) {
        check.valid = false;
        check.message = `Domino ${placement.domino.id} used ${used}x but tray has ${available}x`;
        errors.push(check.message);
      }
    } else if (!puzzle.spec.allowDuplicates) {
      if ((usedCounts.get(placement.domino.id) || 0) > 1) {
        check.valid = false;
        check.message = `Domino ${placement.domino.id} used multiple times`;
        errors.push(check.message);
      }
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
      if (puzzle.spec.regions[r]?.[c] === -1) {
        continue; // hole
      }
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
    const values = cells
      .map((c) => solution.gridPips[c.row][c.col])
      .filter((v): v is number => v !== null && v !== undefined);

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
      const actualSum = values.reduce((a, b) => a + b, 0);
      const ok = (() => {
        switch (constraint.op) {
          case '=':
            return actualSum === constraint.value;
          case '<':
            return actualSum < constraint.value!;
          case '>':
            return actualSum > constraint.value!;
          case '≠':
            return actualSum !== constraint.value;
          default:
            return true;
        }
      })();

      if (!ok) {
        check.valid = false;
        check.message = `Region ${regionId} sum constraint "${constraint.op} ${constraint.value}" violated: sum=${actualSum}`;
        errors.push(check.message);
      } else {
        check.message = `Region ${regionId} sum ${constraint.op} ${constraint.value} ✓ (sum=${actualSum})`;
      }
    }

    // Check all_different constraint
    if (constraint.all_different) {
      const s = new Set(values);
      if (s.size !== values.length) {
        check.valid = false;
        check.message = `Region ${regionId} all_different constraint violated: ${values.join(', ')}`;
        errors.push(check.message);
      } else {
        check.message = `Region ${regionId} all_different ✓`;
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

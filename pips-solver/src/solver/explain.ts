/**
 * Generate human-readable explanations for solver results
 */

import {
  NormalizedPuzzle,
  SolverState,
  Explanation,
  Conflict,
  Cell,
} from '../model/types';
import { getRegionCells } from '../model/normalize';

/**
 * Generate explanation for unsatisfiable puzzle
 */
export function explainUnsatisfiable(
  puzzle: NormalizedPuzzle,
  state: SolverState
): Explanation {
  const conflicts: Conflict[] = [];

  // Check for all_equal regions that need multiple identical doubles
  // This is a common structural issue that should be checked first
  const allEqualConflicts = checkAllEqualDoubleRequirement(puzzle);
  conflicts.push(...allEqualConflicts);

  // Check for empty domains
  const emptyDomains = findEmptyDomains(state);
  if (emptyDomains.length > 0) {
    conflicts.push({
      type: 'no_valid_placement',
      description: `${emptyDomains.length} cell(s) have no valid values remaining`,
      affectedCells: emptyDomains,
    });
  }

  // Check for orphaned cells (cells with no adjacent unfilled cells)
  const orphanedCells = findOrphanedCells(puzzle, state);
  if (orphanedCells.length > 0) {
    conflicts.push({
      type: 'no_valid_placement',
      description: `${orphanedCells.length} cell(s) cannot form domino edges (all neighbors are filled or holes)`,
      affectedCells: orphanedCells,
    });
  }

  // Check for impossible region constraints
  const regionConflicts = findRegionConflicts(puzzle, state);
  conflicts.push(...regionConflicts);

  // Check for domino exhaustion
  const dominoConflict = checkDominoExhaustion(puzzle, state);
  if (dominoConflict) {
    conflicts.push(dominoConflict);
  }

  const details = conflicts.map((c) => {
    let detail = `${c.type}: ${c.description}`;
    if (c.affectedCells) {
      detail += ` (cells: ${c.affectedCells.map((cell) => `(${cell.row},${cell.col})`).join(', ')})`;
    }
    if (c.affectedRegion !== undefined) {
      detail += ` (region ${c.affectedRegion})`;
    }
    return detail;
  });

  return {
    type: 'unsat',
    message: 'Puzzle is unsatisfiable',
    details,
    conflicts,
  };
}

/**
 * Find cells with empty domains
 */
function findEmptyDomains(state: SolverState): Cell[] {
  const cells: Cell[] = [];

  for (const [key, domain] of state.domains) {
    if (domain.length === 0) {
      const [row, col] = key.split(',').map(Number);
      if (state.gridPips[row][col] === null) {
        cells.push({ row, col });
      }
    }
  }

  return cells;
}

/**
 * Find cells that are unfilled but have no unfilled neighbors (orphaned)
 * These cells cannot form domino edges and indicate a dead-end path
 */
function findOrphanedCells(puzzle: NormalizedPuzzle, state: SolverState): Cell[] {
  const orphaned: Cell[] = [];

  for (let row = 0; row < puzzle.spec.rows; row++) {
    for (let col = 0; col < puzzle.spec.cols; col++) {
      // Skip holes
      if (puzzle.spec.regions[row]?.[col] === -1) {
        continue;
      }

      // Skip already filled cells
      if (state.gridPips[row][col] !== null) {
        continue;
      }

      // Check if any adjacent cell is also unfilled
      const hasUnfilledNeighbor = hasAdjacentUnfilledCell(puzzle, state, row, col);
      if (!hasUnfilledNeighbor) {
        orphaned.push({ row, col });
      }
    }
  }

  return orphaned;
}

/**
 * Check if a cell has at least one adjacent unfilled cell
 */
function hasAdjacentUnfilledCell(
  puzzle: NormalizedPuzzle,
  state: SolverState,
  row: number,
  col: number
): boolean {
  const directions = [
    { dr: -1, dc: 0 }, // up
    { dr: 1, dc: 0 },  // down
    { dr: 0, dc: -1 }, // left
    { dr: 0, dc: 1 },  // right
  ];

  for (const { dr, dc } of directions) {
    const newRow = row + dr;
    const newCol = col + dc;

    // Check bounds
    if (newRow < 0 || newRow >= puzzle.spec.rows || newCol < 0 || newCol >= puzzle.spec.cols) {
      continue;
    }

    // Skip holes
    if (puzzle.spec.regions[newRow]?.[newCol] === -1) {
      continue;
    }

    // Check if neighbor is unfilled
    if (state.gridPips[newRow][newCol] === null) {
      return true;
    }
  }

  return false;
}

/**
 * Find region constraint conflicts
 */
function findRegionConflicts(
  puzzle: NormalizedPuzzle,
  state: SolverState
): Conflict[] {
  const conflicts: Conflict[] = [];

  for (const [regionIdStr, constraint] of Object.entries(puzzle.spec.constraints)) {
    const regionId = parseInt(regionIdStr, 10);
    const cells = getRegionCells(puzzle, regionId);

    const assigned: number[] = [];
    const unassigned: Cell[] = [];

    for (const cell of cells) {
      const value = state.gridPips[cell.row][cell.col];
      if (value !== null) {
        assigned.push(value);
      } else {
        unassigned.push(cell);
      }
    }

    // Check sum constraint
    if (constraint.sum !== undefined) {
      const currentSum = assigned.reduce((a, b) => a + b, 0);
      const remaining = constraint.sum - currentSum;

      if (remaining < 0) {
        conflicts.push({
          type: 'region_impossible',
          description: `Region ${regionId} sum already exceeds target (${currentSum} > ${constraint.sum})`,
          affectedRegion: regionId,
          affectedCells: cells,
        });
      }

      if (unassigned.length > 0) {
        const maxPossible = currentSum + unassigned.length * (puzzle.spec.maxPip || 6);
        const minPossible = currentSum + 0;

        if (constraint.sum > maxPossible) {
          conflicts.push({
            type: 'region_impossible',
            description: `Region ${regionId} cannot reach sum ${constraint.sum} (max possible: ${maxPossible})`,
            affectedRegion: regionId,
            affectedCells: cells,
          });
        }

        if (constraint.sum < minPossible) {
          conflicts.push({
            type: 'region_impossible',
            description: `Region ${regionId} will exceed sum ${constraint.sum} (min possible: ${minPossible})`,
            affectedRegion: regionId,
            affectedCells: cells,
          });
        }
      }
    }

    // Check all_equal constraint
    if (constraint.all_equal && assigned.length > 1) {
      const allEqual = assigned.every((v) => v === assigned[0]);
      if (!allEqual) {
        conflicts.push({
          type: 'region_impossible',
          description: `Region ${regionId} all_equal constraint violated (values: ${assigned.join(', ')})`,
          affectedRegion: regionId,
          affectedCells: cells,
        });
      }
    }

    // Check op constraints
    if (constraint.op && constraint.value !== undefined) {
      const violators = assigned.filter((v) => {
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
        conflicts.push({
          type: 'region_impossible',
          description: `Region ${regionId} constraint "${constraint.op} ${constraint.value}" violated by values: ${violators.join(', ')}`,
          affectedRegion: regionId,
          affectedCells: cells,
        });
      }
    }
  }

  return conflicts;
}

/**
 * Check if we've run out of available dominoes
 */
function checkDominoExhaustion(
  puzzle: NormalizedPuzzle,
  state: SolverState
): Conflict | null {
  const trayCount = Array.isArray(puzzle.spec.dominoes) ? puzzle.spec.dominoes.length : 0;
  if (trayCount === 0 && puzzle.spec.allowDuplicates) {
    return null; // Legacy mode: unlimited reuse allowed
  }

  const totalDominoes =
    trayCount > 0 ? trayCount : (((puzzle.spec.maxPip || 6) + 1) * ((puzzle.spec.maxPip || 6) + 2)) / 2;
  const usedCount = state.placements.length;

  // Count how many empty cells there are (each domino covers 2 cells)
  let emptyCells = 0;
  for (let row = 0; row < state.gridPips.length; row++) {
    for (let col = 0; col < state.gridPips[row].length; col++) {
      const regionId = puzzle.spec.regions[row]?.[col];
      // Check if this cell is a valid grid cell (not a hole, i.e. region >= 0) and is empty
      if (regionId !== undefined && regionId !== -1 && state.gridPips[row][col] === null) {
        emptyCells++;
      }
    }
  }

  // Each domino covers 2 cells, so we need emptyCells / 2 dominoes
  const dominoesNeeded = Math.ceil(emptyCells / 2);
  const availableDominoes = totalDominoes - usedCount;

  if (dominoesNeeded > availableDominoes) {
    return {
      type: 'domino_exhausted',
      description: `Not enough dominoes remaining (need ${dominoesNeeded}, have ${availableDominoes})`,
    };
  }

  return null;
}

/**
 * Check for all_equal regions that require multiple identical doubles
 * This is a common cause of unsatisfiable puzzles
 */
function checkAllEqualDoubleRequirement(
  puzzle: NormalizedPuzzle
): Conflict[] {
  const conflicts: Conflict[] = [];
  const tray = puzzle.spec.dominoes || [];

  // Count available doubles in tray
  const doubleCount: Record<number, number> = {};
  for (const [pip1, pip2] of tray) {
    if (pip1 === pip2) {
      doubleCount[pip1] = (doubleCount[pip1] || 0) + 1;
    }
  }

  // Check each all_equal region
  for (const [regionIdStr, constraint] of Object.entries(puzzle.spec.constraints)) {
    if (!constraint.all_equal) continue;

    const regionId = parseInt(regionIdStr, 10);
    const cells = getRegionCells(puzzle, regionId);

    if (cells.length <= 2) continue; // Single domino can handle 2 cells

    // For 4+ cells with all_equal, we need multiple identical dominoes
    // Each domino with both pips equal covers 2 cells with the same value
    const dominoesNeeded = Math.ceil(cells.length / 2);

    if (dominoesNeeded > 1) {
      // Check if any double appears enough times in the tray
      let canSatisfy = false;
      for (const [pip, count] of Object.entries(doubleCount)) {
        if (count >= dominoesNeeded) {
          canSatisfy = true;
          break;
        }
      }

      if (!canSatisfy) {
        conflicts.push({
          type: 'region_impossible',
          description: `Region ${regionId} has ${cells.length} cells with all_equal constraint, requiring ${dominoesNeeded} identical doubles, but tray doesn't have enough matching doubles`,
          affectedRegion: regionId,
          affectedCells: cells,
        });
      }
    }
  }

  return conflicts;
}

/**
 * Generate success explanation with solution summary
 */
export function explainSuccess(puzzle: NormalizedPuzzle, state: SolverState): Explanation {
  const details: string[] = [];

  // Summarize region constraints
  for (const [regionIdStr, constraint] of Object.entries(puzzle.spec.constraints)) {
    const regionId = parseInt(regionIdStr, 10);
    const cells = getRegionCells(puzzle, regionId);
    const values = cells.map((c) => state.gridPips[c.row][c.col]!);

    let constraintDesc = `Region ${regionId}`;

    if (constraint.sum !== undefined) {
      const sum = values.reduce((a, b) => a + b, 0);
      constraintDesc += ` sum=${sum} (target: ${constraint.sum})`;
    }

    if (constraint.all_equal) {
      constraintDesc += ` all_equal (value: ${values[0]})`;
    }

    if (constraint.op && constraint.value !== undefined) {
      const sum = values.reduce((a, b) => a + b, 0);
      constraintDesc += ` sum ${constraint.op} ${constraint.value} (sum: ${sum})`;
    }

    details.push(constraintDesc);
  }

  details.push(`Total dominoes placed: ${state.placements.length}`);

  return {
    type: 'success',
    message: 'Solution found',
    details,
  };
}

/**
 * Generate human-readable explanations for solver results
 */

import {
  NormalizedPuzzle,
  SolverState,
  Explanation,
  Conflict,
  Cell,
  cellKey,
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

  // Check for empty domains
  const emptyDomains = findEmptyDomains(state);
  if (emptyDomains.length > 0) {
    conflicts.push({
      type: 'no_valid_placement',
      description: `${emptyDomains.length} cell(s) have no valid values remaining`,
      affectedCells: emptyDomains,
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
  if (puzzle.spec.allowDuplicates) {
    return null; // Can't exhaust dominoes if duplicates allowed
  }

  const maxPip = puzzle.spec.maxPip || 6;
  const totalDominoes = ((maxPip + 1) * (maxPip + 2)) / 2;
  const usedCount = state.usedDominoes.size;

  // Count how many more dominoes we need
  let emptyEdges = 0;
  for (const edge of puzzle.edges) {
    if (
      state.gridPips[edge.cell1.row][edge.cell1.col] === null &&
      state.gridPips[edge.cell2.row][edge.cell2.col] === null
    ) {
      emptyEdges++;
    }
  }

  const availableDominoes = totalDominoes - usedCount;

  if (emptyEdges > availableDominoes) {
    return {
      type: 'domino_exhausted',
      description: `Not enough dominoes remaining (need ${emptyEdges}, have ${availableDominoes})`,
    };
  }

  return null;
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
      constraintDesc += ` all ${constraint.op} ${constraint.value}`;
    }

    details.push(constraintDesc);
  }

  details.push(`Total dominoes used: ${state.usedDominoes.size}`);

  return {
    type: 'success',
    message: 'Solution found',
    details,
  };
}

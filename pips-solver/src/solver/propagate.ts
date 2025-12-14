/**
 * Constraint propagation and forward checking
 */

import {
  NormalizedPuzzle,
  SolverState,
  Cell,
  cellKey,
  RegionConstraint,
} from '../model/types';
import { getRegionCells, getRegionId } from '../model/normalize';

/**
 * Initialize domains for all cells
 */
export function initializeDomains(
  puzzle: NormalizedPuzzle,
  maxPip: number
): Map<string, number[]> {
  const domains = new Map<string, number[]>();
  const allPips = Array.from({ length: maxPip + 1 }, (_, i) => i);

  for (let row = 0; row < puzzle.spec.rows; row++) {
    for (let col = 0; col < puzzle.spec.cols; col++) {
      const cell: Cell = { row, col };
      domains.set(cellKey(cell), [...allPips]);
    }
  }

  return domains;
}

/**
 * Update domains after assigning a value to a cell
 * Returns false if a domain becomes empty (conflict detected)
 */
export function propagateConstraints(
  puzzle: NormalizedPuzzle,
  state: SolverState,
  assignedCell: Cell,
  assignedValue: number
): boolean {
  // Create a snapshot for potential rollback
  const domainBackup = new Map(state.domains);

  try {
    // Remove assigned value from the cell's domain
    const key = cellKey(assignedCell);
    state.domains.set(key, [assignedValue]);

    // Propagate region constraints
    const regionId = getRegionId(puzzle, assignedCell);
    const constraint = puzzle.spec.constraints[regionId];

    if (constraint) {
      if (!propagateRegionConstraint(puzzle, state, regionId, constraint)) {
        state.domains = domainBackup;
        return false;
      }
    }

    // Check all cells have non-empty domains
    for (const [cellKey, domain] of state.domains) {
      const [row, col] = cellKey.split(',').map(Number);
      if (state.gridPips[row][col] === null && domain.length === 0) {
        state.domains = domainBackup;
        return false;
      }
    }

    return true;
  } catch (error) {
    state.domains = domainBackup;
    return false;
  }
}

/**
 * Propagate constraints within a region
 */
function propagateRegionConstraint(
  puzzle: NormalizedPuzzle,
  state: SolverState,
  regionId: number,
  constraint: RegionConstraint
): boolean {
  const cells = getRegionCells(puzzle, regionId);

  // Get assigned and unassigned cells
  const assigned: number[] = [];
  const unassignedCells: Cell[] = [];

  for (const cell of cells) {
    const value = state.gridPips[cell.row][cell.col];
    if (value !== null) {
      assigned.push(value);
    } else {
      unassignedCells.push(cell);
    }
  }

  // Check sum constraint
  if (constraint.sum !== undefined) {
    const currentSum = assigned.reduce((a, b) => a + b, 0);
    const remaining = constraint.sum - currentSum;

    if (remaining < 0) {
      return false; // Already exceeded sum
    }

    if (unassignedCells.length === 0 && currentSum !== constraint.sum) {
      return false; // All assigned but sum doesn't match
    }

    // Prune domains based on sum constraint
    if (unassignedCells.length > 0) {
      const maxPossible = unassignedCells.length * puzzle.spec.maxPip!;
      if (remaining > maxPossible) {
        return false; // Impossible to reach sum
      }

      const minPossible = 0; // Minimum is all zeros
      if (remaining < minPossible) {
        return false;
      }

      // If only one cell left, it must equal remaining
      if (unassignedCells.length === 1) {
        const cell = unassignedCells[0];
        const key = cellKey(cell);
        const domain = state.domains.get(key) || [];

        if (!domain.includes(remaining)) {
          return false;
        }

        state.domains.set(key, [remaining]);
      }
    }
  }

  // Check all_equal constraint
  if (constraint.all_equal) {
    if (assigned.length > 0) {
      const targetValue = assigned[0];

      // All assigned values must be equal
      if (!assigned.every((v) => v === targetValue)) {
        return false;
      }

      // Restrict unassigned cells to only the target value
      for (const cell of unassignedCells) {
        const key = cellKey(cell);
        const domain = state.domains.get(key) || [];

        if (!domain.includes(targetValue)) {
          return false;
        }

        state.domains.set(key, [targetValue]);
      }
    }
  }

  // Check comparison constraints (op + value)
  if (constraint.op && constraint.value !== undefined) {
    for (const cell of unassignedCells) {
      const key = cellKey(cell);
      const domain = state.domains.get(key) || [];
      const newDomain = domain.filter((pip) =>
        checkConstraintOp(pip, constraint.op!, constraint.value!)
      );

      if (newDomain.length === 0) {
        return false;
      }

      state.domains.set(key, newDomain);
    }

    // Check assigned values satisfy constraint
    for (const value of assigned) {
      if (!checkConstraintOp(value, constraint.op, constraint.value)) {
        return false;
      }
    }
  }

  return true;
}

/**
 * Check if a value satisfies an operator constraint
 */
function checkConstraintOp(value: number, op: string, target: number): boolean {
  switch (op) {
    case '=':
      return value === target;
    case '<':
      return value < target;
    case '>':
      return value > target;
    case '≠':
      return value !== target;
    default:
      return true;
  }
}

/**
 * Deep copy domains map
 */
export function copyDomains(domains: Map<string, number[]>): Map<string, number[]> {
  const copy = new Map<string, number[]>();
  for (const [key, values] of domains) {
    copy.set(key, [...values]);
  }
  return copy;
}

/**
 * Check if current state is consistent with all constraints
 */
export function isConsistent(puzzle: NormalizedPuzzle, state: SolverState): boolean {
  // Check all region constraints
  for (const [regionIdStr, constraint] of Object.entries(puzzle.spec.constraints)) {
    const regionId = parseInt(regionIdStr, 10);
    const cells = getRegionCells(puzzle, regionId);

    const assigned: number[] = [];
    let hasUnassigned = false;

    for (const cell of cells) {
      const value = state.gridPips[cell.row][cell.col];
      if (value !== null) {
        assigned.push(value);
      } else {
        hasUnassigned = true;
      }
    }

    // Only check constraints if all cells are assigned
    if (!hasUnassigned) {
      if (constraint.sum !== undefined) {
        const sum = assigned.reduce((a, b) => a + b, 0);
        if (sum !== constraint.sum) {
          return false;
        }
      }

      if (constraint.all_equal) {
        if (!assigned.every((v) => v === assigned[0])) {
          return false;
        }
      }

      if (constraint.op && constraint.value !== undefined) {
        if (!assigned.every((v) => checkConstraintOp(v, constraint.op!, constraint.value!))) {
          return false;
        }
      }
    }
  }

  return true;
}

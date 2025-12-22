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

  // Default: all pips for all non-hole cells
  for (let row = 0; row < puzzle.spec.rows; row++) {
    for (let col = 0; col < puzzle.spec.cols; col++) {
      if (puzzle.spec.regions[row]?.[col] === -1) continue; // hole
      const cell: Cell = { row, col };
      domains.set(cellKey(cell), allPips);
    }
  }

  // Initial pruning based on region constraints (cheap and effective)
  for (const [regionIdStr, constraint] of Object.entries(puzzle.spec.constraints)) {
    const regionId = parseInt(regionIdStr, 10);
    if (Number.isNaN(regionId)) continue;

    const cells = getRegionCells(puzzle, regionId);
    if (cells.length === 0) continue;

    // all_different feasibility (quick fail)
    if (constraint.all_different && cells.length > maxPip + 1) {
      // Force an immediate contradiction: empty one domain.
      domains.set(cellKey(cells[0]), []);
      continue;
    }

    const hasEqSum = constraint.sum !== undefined || (constraint.op === '=' && constraint.value !== undefined);
    const targetSum = constraint.sum !== undefined ? constraint.sum : (constraint.op === '=' ? constraint.value : undefined);

    // If we know the exact sum, we can bound each cell by simple arithmetic (non-negative pips).
    if (hasEqSum && targetSum !== undefined) {
      const n = cells.length;

      // If all_equal + exact sum, it's forced.
      if (constraint.all_equal) {
        if (targetSum % n !== 0) {
          domains.set(cellKey(cells[0]), []);
          continue;
        }
        const v = targetSum / n;
        if (v < 0 || v > maxPip) {
          domains.set(cellKey(cells[0]), []);
          continue;
        }
        for (const cell of cells) {
          domains.set(cellKey(cell), [v]);
        }
        continue;
      }

      const lo = Math.max(0, targetSum - (n - 1) * maxPip);
      const hi = Math.min(maxPip, targetSum);
      const bounded = allPips.filter((p) => p >= lo && p <= hi);
      for (const cell of cells) {
        domains.set(cellKey(cell), bounded);
      }
      continue;
    }

    // If sum < K, every cell must be < K (since pips are non-negative).
    if (constraint.op === '<' && constraint.value !== undefined) {
      const hi = Math.min(maxPip, constraint.value - 1);
      const bounded = allPips.filter((p) => p <= hi);
      for (const cell of cells) {
        domains.set(cellKey(cell), bounded);
      }
      continue;
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
  // Should never assign into a hole, but be defensive.
  if (puzzle.spec.regions[assignedCell.row]?.[assignedCell.col] === -1) {
    return false;
  }

  // NOTE: We intentionally do not snapshot/rollback domains here.
  // The solver already snapshots full state per placement and will restore on backtrack.

  // Remove assigned value from the cell's domain
  const key = cellKey(assignedCell);
  state.domains.set(key, [assignedValue]);

  // Propagate region constraints
  const regionId = getRegionId(puzzle, assignedCell);
  const constraint = puzzle.spec.constraints[regionId];

  if (constraint) {
    if (!propagateRegionConstraint(puzzle, state, regionId, constraint)) {
      return false;
    }
  }

  // Check all cells have non-empty domains
  for (const [cellKey, domain] of state.domains) {
    const [row, col] = cellKey.split(',').map(Number);
    if (puzzle.spec.regions[row]?.[col] === -1) {
      continue; // hole
    }
    if (state.gridPips[row][col] === null && domain.length === 0) {
      return false;
    }
  }

  // Note: Orphan detection is now handled by the cell-first solver approach
  // The solver picks cells first and pairs with neighbors, so orphans are
  // naturally detected when a cell has no unfilled neighbors to pair with.
  // We keep a lightweight check here for early pruning.
  const orphanResult = checkNoOrphanedCells(puzzle, state);
  if (!orphanResult) {
    return false;
  }

  return true;
}

/**
 * Check that no unfilled cell is orphaned (has no adjacent unfilled cells)
 * Returns false if any orphaned cell is found
 *
 * Note: With the cell-first solver approach, this check is mostly redundant
 * but kept for early pruning during constraint propagation.
 */
function checkNoOrphanedCells(
  puzzle: NormalizedPuzzle,
  state: SolverState
): boolean {
  let unfilledCount = 0;

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

      unfilledCount++;

      // Check if this cell has at least one adjacent unfilled cell
      let hasUnfilledNeighbor = false;
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
          hasUnfilledNeighbor = true;
          break;
        }
      }

      // If no unfilled neighbor, this cell is orphaned - dead end
      if (!hasUnfilledNeighbor) {
        return false;
      }
    }
  }

  // Check for odd number of unfilled cells (can't pair them all into dominoes)
  if (unfilledCount % 2 !== 0) {
    return false;
  }

  return true;
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

  // Check all_different constraint
  if (constraint.all_different) {
    const seen = new Set<number>();
    for (const v of assigned) {
      if (seen.has(v)) {
        return false;
      }
      seen.add(v);
    }
    // Forward check: remove assigned values from unassigned domains
    for (const cell of unassignedCells) {
      const key = cellKey(cell);
      const domain = state.domains.get(key) || [];
      const newDomain = domain.filter((pip) => !seen.has(pip));
      if (newDomain.length === 0) {
        return false;
      }
      state.domains.set(key, newDomain);
    }
  }

  // Region sum constraints:
  // - `sum`: exact equality
  // - `op/value`: inequality on the REGION SUM (not per-cell)
  const currentSum = assigned.reduce((a, b) => a + b, 0);

  // Compute tight bounds from domains for remaining cells.
  let minPossibleSum = currentSum;
  let maxPossibleSum = currentSum;
  for (const cell of unassignedCells) {
    const key = cellKey(cell);
    const domain = state.domains.get(key) || [];
    if (domain.length === 0) {
      return false;
    }
    minPossibleSum += Math.min(...domain);
    maxPossibleSum += Math.max(...domain);
  }

  // Exact sum
  if (constraint.sum !== undefined) {
    const target = constraint.sum;
    if (target < minPossibleSum || target > maxPossibleSum) {
      return false;
    }

    if (unassignedCells.length === 0 && currentSum !== target) {
      return false;
    }

    // If only one cell left, force it.
    if (unassignedCells.length === 1) {
      const cell = unassignedCells[0];
      const key = cellKey(cell);
      const domain = state.domains.get(key) || [];
      const forced = target - currentSum;
      if (!domain.includes(forced)) {
        return false;
      }
      state.domains.set(key, [forced]);
    }
  }

  // Sum inequality (op/value)
  if (constraint.op && constraint.value !== undefined) {
    const op = constraint.op;
    const target = constraint.value;

    // Conservative feasibility check using min/max possible sums.
    if (op === '=') {
      if (target < minPossibleSum || target > maxPossibleSum) {
        return false;
      }
      if (unassignedCells.length === 0 && currentSum !== target) {
        return false;
      }
    } else if (op === '<') {
      if (minPossibleSum >= target) {
        return false;
      }
      if (unassignedCells.length === 0 && !(currentSum < target)) {
        return false;
      }
    } else if (op === '>') {
      if (maxPossibleSum <= target) {
        return false;
      }
      if (unassignedCells.length === 0 && !(currentSum > target)) {
        return false;
      }
    } else if (op === '≠') {
      if (unassignedCells.length === 0 && currentSum === target) {
        return false;
      }
      // If the sum is forced to equal target regardless of assignments, prune.
      if (minPossibleSum === maxPossibleSum && minPossibleSum === target) {
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
  // We only ever replace domain arrays (never mutate them in place),
  // so a shallow copy is safe and much faster than deep cloning.
  return new Map(domains);
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
      const sum = assigned.reduce((a, b) => a + b, 0);

      if (constraint.sum !== undefined && sum !== constraint.sum) {
        return false;
      }

      if (constraint.op && constraint.value !== undefined) {
        if (!checkConstraintOp(sum, constraint.op, constraint.value)) {
          return false;
        }
      }

      if (constraint.all_equal) {
        if (!assigned.every((v) => v === assigned[0])) {
          return false;
        }
      }

      if (constraint.all_different) {
        const s = new Set(assigned);
        if (s.size !== assigned.length) {
          return false;
        }
      }
    }
  }

  return true;
}

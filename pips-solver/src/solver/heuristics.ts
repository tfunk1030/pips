/**
 * Heuristics for CSP solving
 * Implements MRV (Minimum Remaining Values) and other ordering strategies
 *
 * Uses cell-first approach (like Python solver) instead of edge-first.
 * This prevents orphaned cells by always picking a cell and pairing with a neighbor.
 */

import { NormalizedPuzzle, SolverState, Edge, Cell, cellKey, Domino, dominoId } from '../model/types';
import { getNeighbors } from '../model/normalize';

/**
 * Choose the next cell to fill using MRV heuristic
 * Picks an unfilled cell with fewest unfilled neighbors (most constrained first)
 * This matches the Python solver's pick_next_cell algorithm
 */
export function selectNextCell(
  puzzle: NormalizedPuzzle,
  state: SolverState
): Cell | null {
  let bestCell: Cell | null = null;
  let minUnfilledNeighbors = Infinity;

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

      // Count unfilled neighbors
      const cell: Cell = { row, col };
      const neighbors = getNeighbors(puzzle, cell);
      const unfilledNeighbors = neighbors.filter(
        n => state.gridPips[n.row][n.col] === null
      );

      // MRV: prefer cells with fewer unfilled neighbors
      // This helps prune the search space earlier
      if (unfilledNeighbors.length < minUnfilledNeighbors) {
        minUnfilledNeighbors = unfilledNeighbors.length;
        bestCell = cell;
      }
    }
  }

  return bestCell;
}

/**
 * Get unfilled neighbors for a cell
 * Returns cells that can be paired with to form a domino edge
 */
export function getUnfilledNeighborsForCell(
  puzzle: NormalizedPuzzle,
  state: SolverState,
  cell: Cell
): Cell[] {
  const neighbors = getNeighbors(puzzle, cell);
  return neighbors.filter(n => state.gridPips[n.row][n.col] === null);
}

/**
 * Choose the next edge to fill using MRV heuristic
 * Picks the edge with the most constrained cells
 * @deprecated Use selectNextCell + getUnfilledNeighborsForCell instead
 */
export function selectNextEdge(
  puzzle: NormalizedPuzzle,
  state: SolverState
): Edge | null {
  let bestEdge: Edge | null = null;
  let minDomainProduct = Infinity;

  for (const edge of puzzle.edges) {
    const key1 = cellKey(edge.cell1);
    const key2 = cellKey(edge.cell2);

    // Skip if either cell is already filled
    if (state.gridPips[edge.cell1.row][edge.cell1.col] !== null ||
        state.gridPips[edge.cell2.row][edge.cell2.col] !== null) {
      continue;
    }

    // Get domain sizes
    const domain1 = state.domains.get(key1) || [];
    const domain2 = state.domains.get(key2) || [];

    if (domain1.length === 0 || domain2.length === 0) {
      // Dead end - return this edge to trigger backtrack
      return edge;
    }

    // MRV: prefer edges with smaller domain product
    const domainProduct = domain1.length * domain2.length;

    if (domainProduct < minDomainProduct) {
      minDomainProduct = domainProduct;
      bestEdge = edge;
    }
  }

  return bestEdge;
}

/**
 * Generate candidate dominoes for an edge
 * Returns dominoes that could be placed on this edge
 */
export function getCandidateDominoes(
  puzzle: NormalizedPuzzle,
  edge: Edge,
  state: SolverState,
  maxPip: number,
  allowDuplicates: boolean
): Domino[] {
  const key1 = cellKey(edge.cell1);
  const key2 = cellKey(edge.cell2);

  const domain1 = state.domains.get(key1) || [];
  const domain2 = state.domains.get(key2) || [];

  const candidates: Domino[] = [];

  // Build inventory counts if the puzzle specifies an explicit tray.
  // NOTE: This is still per-call; the next optimization step is to precompute this once.
  const inventoryCounts = new Map<string, number>();
  if (Array.isArray(puzzle.spec.dominoes) && puzzle.spec.dominoes.length > 0) {
    for (const [a, b] of puzzle.spec.dominoes) {
      const id = dominoId(a, b);
      inventoryCounts.set(id, (inventoryCounts.get(id) || 0) + 1);
    }
  }

  // Track which (pip1, pip2) combinations we've already added to avoid duplicates
  const seen = new Set<string>();

  for (const pip1 of domain1) {
    for (const pip2 of domain2) {
      const id = dominoId(pip1, pip2);

      // If we have an explicit tray inventory, enforce multiset counts.
      if (inventoryCounts.size > 0) {
        const available = inventoryCounts.get(id) || 0;
        const used = state.usedDominoes.get(id) || 0;
        if (used >= available) {
          continue;
        }
      } else {
        // Legacy behavior: if duplicates aren't allowed, each id can be used at most once.
        if (!allowDuplicates) {
          const used = state.usedDominoes.get(id) || 0;
          if (used >= 1) {
            continue;
          }
        }
      }

      // Add this orientation (pip1 at cell1, pip2 at cell2)
      const key1 = `${pip1}-${pip2}`;
      if (!seen.has(key1)) {
        seen.add(key1);
        candidates.push({
          id,
          pip1,
          pip2,
        });
      }

      // For non-doubles, also try the flipped orientation (pip2 at cell1, pip1 at cell2)
      // This is critical: a domino [3,4] can be placed as (3,4) or (4,3) on an edge
      if (pip1 !== pip2 && domain1.includes(pip2) && domain2.includes(pip1)) {
        const key2 = `${pip2}-${pip1}`;
        if (!seen.has(key2)) {
          seen.add(key2);
          candidates.push({
            id,
            pip1: pip2,
            pip2: pip1,
          });
        }
      }
    }
  }

  return candidates;
}

/**
 * Order dominoes by constraint (prefer more constrained first)
 */
export function orderDominoes(dominoes: Domino[], state: SolverState): Domino[] {
  // For now, use the order as-is
  // Could enhance with domain filtering, constraint checking, etc.
  return dominoes;
}

/**
 * Get unassigned neighbors of a cell
 */
export function getUnassignedNeighbors(
  puzzle: NormalizedPuzzle,
  cell: Cell,
  state: SolverState
): Cell[] {
  const neighbors = getNeighbors(puzzle, cell);
  return neighbors.filter(
    (n) => state.gridPips[n.row][n.col] === null
  );
}

/**
 * Count constraints on a cell (for heuristics)
 */
export function countCellConstraints(
  puzzle: NormalizedPuzzle,
  cell: Cell,
  state: SolverState
): number {
  const regionId = puzzle.spec.regions[cell.row][cell.col];
  const constraint = puzzle.spec.constraints[regionId];

  let count = 0;

  if (constraint) {
    if (constraint.sum !== undefined) count++;
    if (constraint.op !== undefined) count++;
    if (constraint.all_equal) count++;
  }

  // Add constraints from unassigned neighbors
  count += getUnassignedNeighbors(puzzle, cell, state).length;

  return count;
}

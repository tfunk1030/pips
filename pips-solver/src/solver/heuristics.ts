/**
 * Heuristics for CSP solving
 * Implements MRV (Minimum Remaining Values) and other ordering strategies
 */

import { NormalizedPuzzle, SolverState, Edge, Cell, cellKey, Domino } from '../model/types';
import { getNeighbors } from '../model/normalize';

/**
 * Choose the next edge to fill using MRV heuristic
 * Picks the edge with the most constrained cells
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
  const seen = new Set<string>();

  for (const pip1 of domain1) {
    for (const pip2 of domain2) {
      // Create domino ID (normalized)
      const id = pip1 <= pip2 ? `${pip1}-${pip2}` : `${pip2}-${pip1}`;

      // Skip if we've already considered this domino from the other orientation
      if (seen.has(id)) {
        continue;
      }
      seen.add(id);

      // Check if domino is available
      if (!allowDuplicates && state.usedDominoes.has(id)) {
        continue;
      }

      candidates.push({
        id,
        pip1,
        pip2,
      });
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

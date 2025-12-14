import type { Puzzle } from '../model/types';
import type { SolverState } from './solver';

/**
 * MRV: pick the next unpaired cell with the fewest legal moves.
 * Deterministic tie-breaker: lowest cellId.
 */
export function pickNextCellMRV(p: Puzzle, st: SolverState, moveCountForCell: (cellId: number, bestSoFar: number) => number): number {
  let bestCell = -1;
  let bestCount = Number.POSITIVE_INFINITY;
  for (let cellId = 0; cellId < p.cells.length; cellId++) {
    if (st.mate[cellId] !== -1) continue;
    const cnt = moveCountForCell(cellId, bestCount);
    if (cnt === 0) return cellId; // immediate dead-end
    if (cnt < bestCount) {
      bestCount = cnt;
      bestCell = cellId;
      if (bestCount === 1) break;
    }
  }
  return bestCell;
}




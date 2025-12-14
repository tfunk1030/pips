import type { Puzzle, RegionConstraint } from '../model/types';
import type { RegionState } from './propagate';

export function explainNoMoves(p: Puzzle, cellId: number): string {
  const cell = p.cells[cellId];
  return `No valid domino placements for cell (${cell.r},${cell.c}).`;
}

export function explainRegionInfeasible(p: Puzzle, regionId: string, regionSize: number, st: RegionState, c: RegionConstraint, reason?: string): string {
  const op = c.sum != null ? '=' : c.op;
  const val = c.sum != null ? c.sum : c.value;
  const sumPart = op && val != null ? `sum ${op} ${val}` : 'no sum constraint';
  const aePart = c.all_equal ? ', all_equal' : '';
  const base = `Region ${regionId} infeasible (${sumPart}${aePart}). assignedSum=${st.assignedSum} unassigned=${st.unassigned}`;
  return reason ? `${base}; ${reason}.` : `${base}.`;
}

export function explainDominoExhausted(dominoKey: string): string {
  return `Domino ${dominoKey} exhausted.`;
}




import { NormalizedPuzzle } from '../model/normalize';
import { DomainMap, Variable } from './types';

export function selectNextVariable(puzzle: NormalizedPuzzle, domains: DomainMap, assignment: Map<string, number>): Variable | null {
  let best: Variable | null = null;
  let bestSize = Infinity;
  for (let r = 0; r < puzzle.rows; r++) {
    for (let c = 0; c < puzzle.cols; c++) {
      const key = `${r},${c}`;
      if (assignment.has(key)) continue;
      const size = domains[key].length;
      if (size < bestSize) {
        bestSize = size;
        best = { r, c };
      }
    }
  }
  return best;
}

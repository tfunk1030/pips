import { PuzzleSpec } from './types';

export interface NormalizedPuzzle extends PuzzleSpec {
  adjacency: [number, number][][][]; // adjacency[r][c] -> list of [r,c]
  regionCells: Record<number, [number, number][]>;
}

export function normalizePuzzle(puzzle: PuzzleSpec): NormalizedPuzzle {
  const adjacency: [number, number][][][] = [];
  const regionCells: Record<number, [number, number][]> = {};
  for (let r = 0; r < puzzle.rows; r++) {
    adjacency[r] = [];
    for (let c = 0; c < puzzle.cols; c++) {
      const cellRegion = puzzle.regions[r][c];
      if (!regionCells[cellRegion]) regionCells[cellRegion] = [];
      regionCells[cellRegion].push([r, c]);
      const neighbors: [number, number][] = [];
      if (r > 0) neighbors.push([r - 1, c]);
      if (r < puzzle.rows - 1) neighbors.push([r + 1, c]);
      if (c > 0) neighbors.push([r, c - 1]);
      if (c < puzzle.cols - 1) neighbors.push([r, c + 1]);
      adjacency[r][c] = neighbors;
    }
  }
  return { ...puzzle, adjacency, regionCells };
}

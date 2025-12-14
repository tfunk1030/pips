/**
 * Normalize puzzle specifications for efficient solving
 * Precomputes adjacency lists, region cells, and all valid edges
 */

import { PuzzleSpec, NormalizedPuzzle, Cell, Edge, cellKey } from './types';

/**
 * Normalize a puzzle specification
 */
export function normalizePuzzle(spec: PuzzleSpec): NormalizedPuzzle {
  const adjacencyList = buildAdjacencyList(spec);
  const regionCells = buildRegionCells(spec);
  const edges = buildEdges(spec);

  return {
    spec,
    adjacencyList,
    regionCells,
    edges,
  };
}

/**
 * Build adjacency list for all cells
 */
function buildAdjacencyList(spec: PuzzleSpec): Map<string, Cell[]> {
  const adjacencyList = new Map<string, Cell[]>();

  for (let row = 0; row < spec.rows; row++) {
    for (let col = 0; col < spec.cols; col++) {
      const cell: Cell = { row, col };
      const key = cellKey(cell);
      const neighbors: Cell[] = [];

      // Check all 4 orthogonal neighbors
      const directions = [
        { dr: -1, dc: 0 }, // up
        { dr: 1, dc: 0 },  // down
        { dr: 0, dc: -1 }, // left
        { dr: 0, dc: 1 },  // right
      ];

      for (const { dr, dc } of directions) {
        const newRow = row + dr;
        const newCol = col + dc;

        if (newRow >= 0 && newRow < spec.rows && newCol >= 0 && newCol < spec.cols) {
          neighbors.push({ row: newRow, col: newCol });
        }
      }

      adjacencyList.set(key, neighbors);
    }
  }

  return adjacencyList;
}

/**
 * Build mapping of region IDs to cells
 */
function buildRegionCells(spec: PuzzleSpec): Map<number, Cell[]> {
  const regionCells = new Map<number, Cell[]>();

  for (let row = 0; row < spec.rows; row++) {
    for (let col = 0; col < spec.cols; col++) {
      const regionId = spec.regions[row][col];

      if (!regionCells.has(regionId)) {
        regionCells.set(regionId, []);
      }

      regionCells.get(regionId)!.push({ row, col });
    }
  }

  return regionCells;
}

/**
 * Build all possible domino edges (pairs of adjacent cells)
 */
function buildEdges(spec: PuzzleSpec): Edge[] {
  const edges: Edge[] = [];
  const seenEdges = new Set<string>();

  for (let row = 0; row < spec.rows; row++) {
    for (let col = 0; col < spec.cols; col++) {
      const cell1: Cell = { row, col };

      // Only check right and down to avoid duplicates
      const neighbors = [
        { row: row, col: col + 1 },     // right
        { row: row + 1, col: col },     // down
      ];

      for (const cell2 of neighbors) {
        if (cell2.row < spec.rows && cell2.col < spec.cols) {
          // Create a canonical edge key (sorted by row,col)
          const key1 = cellKey(cell1);
          const key2 = cellKey(cell2);
          const edgeKey = key1 < key2 ? `${key1}|${key2}` : `${key2}|${key1}`;

          if (!seenEdges.has(edgeKey)) {
            seenEdges.add(edgeKey);
            edges.push({ cell1, cell2 });
          }
        }
      }
    }
  }

  return edges;
}

/**
 * Get cells in a specific region
 */
export function getRegionCells(normalized: NormalizedPuzzle, regionId: number): Cell[] {
  return normalized.regionCells.get(regionId) || [];
}

/**
 * Get neighbors of a cell
 */
export function getNeighbors(normalized: NormalizedPuzzle, cell: Cell): Cell[] {
  return normalized.adjacencyList.get(cellKey(cell)) || [];
}

/**
 * Get region ID for a cell
 */
export function getRegionId(normalized: NormalizedPuzzle, cell: Cell): number {
  return normalized.spec.regions[cell.row][cell.col];
}

/**
 * Check if two cells are in the same region
 */
export function inSameRegion(normalized: NormalizedPuzzle, cell1: Cell, cell2: Cell): boolean {
  return getRegionId(normalized, cell1) === getRegionId(normalized, cell2);
}

/**
 * Get all region IDs in the puzzle
 */
export function getAllRegionIds(normalized: NormalizedPuzzle): number[] {
  return Array.from(normalized.regionCells.keys()).sort((a, b) => a - b);
}

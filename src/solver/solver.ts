import { normalizePuzzle, NormalizedPuzzle } from '../model/normalize';
import { PuzzleSpec, SolveResult, SolverConfig, SolverProgress, SolutionGrid, SolverStats } from '../model/types';
import { selectNextVariable } from './heuristics';
import { cloneDomains, propagateRegionBounds } from './propagate';
import { formatConflict } from './explain';
import { DomainMap } from './types';
import { validateSolution } from '../validator/validateSolution';
import { validateSpec } from '../validator/validateSpec';

function initialDomains(puzzle: PuzzleSpec): DomainMap {
  const domain = Array.from({ length: (puzzle.maxPip ?? 6) + 1 }).map((_, i) => i);
  const map: DomainMap = {};
  for (let r = 0; r < puzzle.rows; r++) {
    for (let c = 0; c < puzzle.cols; c++) {
      map[`${r},${c}`] = [...domain];
    }
  }
  return map;
}

function sleep(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function buildDominoes(puzzle: NormalizedPuzzle, grid: number[][]): { dominoes: SolutionGrid['dominoes']; conflict?: string } {
  const used: Record<string, boolean> = {};
  const dominoes: SolutionGrid['dominoes'] = [];
  for (let r = 0; r < puzzle.rows; r++) {
    for (let c = 0; c < puzzle.cols; c++) {
      const key = `${r},${c}`;
      if (used[key]) continue;
      const neighbors = puzzle.adjacency[r][c];
      const neighbor = neighbors.find(([nr, nc]) => !used[`${nr},${nc}`]);
      if (!neighbor) {
        return { dominoes, conflict: `Cell ${key} has no free neighbor for domino coverage` };
      }
      const [nr, nc] = neighbor;
      used[key] = true;
      used[`${nr},${nc}`] = true;
      const values: [number, number] = [grid[r][c], grid[nr][nc]];
      const pairId = values[0] <= values[1] ? `${values[0]}-${values[1]}` : `${values[1]}-${values[0]}`;
      const count = dominoes.filter((d) => d.id === pairId).length;
      const maxAllowed = puzzle.allowDuplicates ? Infinity : 1;
      if (count >= maxAllowed) {
        return { dominoes, conflict: `Domino ${pairId} exhausted` };
      }
      dominoes.push({ id: pairId, cells: [[r, c], [nr, nc]], values });
    }
  }
  return { dominoes };
}

async function search(
  puzzle: NormalizedPuzzle,
  assignment: Map<string, number>,
  domains: DomainMap,
  stats: SolverStats,
  config: SolverConfig,
  onProgress: (p: SolverProgress) => void,
  tickCounter: { value: number },
): Promise<{ solution?: SolutionGrid; conflict?: string }> {
  if (assignment.size === puzzle.rows * puzzle.cols) {
    const grid: number[][] = Array.from({ length: puzzle.rows }, () => Array(puzzle.cols).fill(0));
    assignment.forEach((value, key) => {
      const [r, c] = key.split(',').map(Number);
      grid[r][c] = value;
    });
    const { dominoes, conflict } = buildDominoes(puzzle, grid);
    if (conflict) {
      return { conflict };
    }
    return { solution: { gridPips: grid, dominoes } };
  }

  const variable = selectNextVariable(puzzle, domains, assignment);
  if (!variable) return { conflict: 'No variable found' };
  const key = `${variable.r},${variable.c}`;
  const domain = [...domains[key]];
  for (const value of domain) {
    stats.nodes += 1;
    assignment.set(key, value);
    const clonedDomains = cloneDomains(domains);
    clonedDomains[key] = [value];
    const result = propagateRegionBounds(puzzle, clonedDomains, assignment);
    stats.prunes += result.pruned;
    if (result.conflict) {
      stats.backtracks += 1;
      assignment.delete(key);
      continue;
    }
    tickCounter.value += 1;
    if (tickCounter.value % config.progressInterval === 0) {
      onProgress({ nodes: stats.nodes, backtracks: stats.backtracks });
      await sleep();
    }
    const deeper = await search(puzzle, assignment, result.domains, stats, config, onProgress, tickCounter);
    if (deeper.solution) return deeper;
    assignment.delete(key);
    stats.backtracks += 1;
  }
  return { conflict: 'Exhausted domain' };
}

export async function solvePuzzle(
  puzzle: PuzzleSpec,
  config: SolverConfig,
  onProgress: (p: SolverProgress) => void,
): Promise<SolveResult> {
  const specReport = validateSpec(puzzle);
  if (!specReport.ok) {
    return {
      status: 'unsat',
      stats: { nodes: 0, backtracks: 0, prunes: 0, elapsedMs: 0 },
      explanation: 'Specification invalid',
      validationReport: specReport,
    };
  }
  const normalized = normalizePuzzle(puzzle);
  const domains = initialDomains(puzzle);
  const assignment = new Map<string, number>();
  const stats: SolverStats = { nodes: 0, backtracks: 0, prunes: 0, elapsedMs: 0 };
  const start = Date.now();
  const tickCounter = { value: 0 };
  const result = await search(normalized, assignment, domains, stats, config, onProgress, tickCounter);
  stats.elapsedMs = Date.now() - start;
  if (result.solution) {
    const validationReport = validateSolution(puzzle, result.solution);
    return { status: 'solved', solution: result.solution, stats, validationReport };
  }
  return {
    status: 'unsat',
    explanation: formatConflict(result.conflict ?? 'Unknown'),
    stats,
    validationReport: { ok: false, issues: [{ level: 'error', message: result.conflict ?? 'Unknown conflict' }] },
  };
}

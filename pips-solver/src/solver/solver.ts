/**
 * Main CSP solver using backtracking with constraint propagation
 */

import {
  DominoPlacement,
  Explanation,
  NormalizedPuzzle,
  Solution,
  SolverConfig,
  SolverProgress,
  SolverState,
} from '../model/types';
import { explainSuccess, explainUnsatisfiable } from './explain';
import { getCandidateDominoes, selectNextCell, getUnfilledNeighborsForCell, selectNextEdge } from './heuristics';
import { copyDomains, initializeDomains, isConsistent, propagateConstraints } from './propagate';

export interface SolverResult {
  success: boolean;
  solutions: Solution[];
  explanation: Explanation;
  stats: {
    nodes: number;
    backtracks: number;
    prunes: number;
    timeMs: number;
  };
}

/**
 * Solve a puzzle using CSP backtracking
 */
export function solvePuzzle(puzzle: NormalizedPuzzle, config: SolverConfig): SolverResult {
  const startTime = Date.now();

  // Initialize solver state
  const initialState: SolverState = {
    gridPips: Array(puzzle.spec.rows)
      .fill(null)
      .map(() => Array(puzzle.spec.cols).fill(null)),
    usedDominoes: new Map(),
    placements: [],
    domains: initializeDomains(puzzle, config.maxPip),
  };

  const stats = {
    nodes: 0,
    backtracks: 0,
    prunes: 0,
    timeMs: 0,
  };

  const solutions: Solution[] = [];

  // Run backtracking search
  backtrack(puzzle, initialState, config, stats, solutions);

  stats.timeMs = Date.now() - startTime;
  // Stamp time on solutions (solution objects are created during search)
  for (const sol of solutions) {
    sol.stats.timeMs = stats.timeMs;
    sol.stats.solutions = solutions.length;
  }

  if (solutions.length > 0) {
    return {
      success: true,
      solutions,
      explanation: explainSuccess(
        puzzle,
        solutions[0].dominoes.length > 0 ? stateFromSolution(solutions[0], puzzle) : initialState
      ),
      stats,
    };
  } else {
    return {
      success: false,
      solutions: [],
      explanation: explainUnsatisfiable(puzzle, initialState),
      stats,
    };
  }
}

/**
 * Recursive backtracking with constraint propagation
 * Uses CELL-FIRST approach (like Python solver) to prevent orphaned cells
 */
function backtrack(
  puzzle: NormalizedPuzzle,
  state: SolverState,
  config: SolverConfig,
  stats: { nodes: number; backtracks: number; prunes: number },
  solutions: Solution[]
): boolean {
  stats.nodes++;

  // Check if we have a complete solution
  if (isComplete(state, puzzle)) {
    if (isConsistent(puzzle, state)) {
      const solution: Solution = {
        gridPips: state.gridPips.map(row => [...row]),
        dominoes: [...state.placements],
        stats: {
          nodes: stats.nodes,
          backtracks: stats.backtracks,
          prunes: stats.prunes,
          timeMs: 0,
          solutions: solutions.length + 1,
        },
      };

      solutions.push(solution);

      if (config.debugLevel >= 1) {
        console.log(`Found solution ${solutions.length}`);
      }

      // If finding all solutions, continue; otherwise stop
      return !config.findAll;
    }
  }

  // CELL-FIRST APPROACH: Pick a cell, then pair it with each unfilled neighbor
  // This prevents orphaned cells by always considering cell-neighbor pairs
  const cell = selectNextCell(puzzle, state);

  if (!cell) {
    // No more unfilled cells - but we already checked isComplete, so this is a bug
    return false;
  }

  // Get unfilled neighbors to pair with
  const neighbors = getUnfilledNeighborsForCell(puzzle, state, cell);

  if (neighbors.length === 0) {
    // Cell has no unfilled neighbors - this is an orphan!
    stats.prunes++;
    if (config.debugLevel >= 2) {
      console.log(`Cell (${cell.row},${cell.col}) is orphaned - no unfilled neighbors`);
    }
    return false;
  }

  // Try each neighbor
  for (const neighbor of neighbors) {
    // Create edge for this cell-neighbor pair
    const edge = { cell1: cell, cell2: neighbor };

    // Get candidate dominoes for this edge
    const candidates = getCandidateDominoes(
      puzzle,
      edge,
      state,
      config.maxPip,
      config.allowDuplicates
    );

    if (candidates.length === 0) {
      // No valid dominoes for this edge - try next neighbor
      continue;
    }

    // Try each candidate domino
    for (const domino of candidates) {
      // Create placement
      const placement: DominoPlacement = {
        domino,
        cell1: edge.cell1,
        cell2: edge.cell2,
      };

      // Save state for backtracking
      const savedDomains = copyDomains(state.domains);
      const savedUsedDominoes = new Map(state.usedDominoes);

      // Apply placement
      state.gridPips[edge.cell1.row][edge.cell1.col] = domino.pip1;
      state.gridPips[edge.cell2.row][edge.cell2.col] = domino.pip2;
      state.usedDominoes.set(domino.id, (state.usedDominoes.get(domino.id) || 0) + 1);
      state.placements.push(placement);

      // Propagate constraints
      const prop1 = propagateConstraints(puzzle, state, edge.cell1, domino.pip1);
      const prop2 = prop1 && propagateConstraints(puzzle, state, edge.cell2, domino.pip2);

      if (prop1 && prop2) {
        // Recurse
        if (config.debugLevel >= 2) {
          console.log(
            `Trying domino ${domino.id} at (${edge.cell1.row},${edge.cell1.col})-(${edge.cell2.row},${edge.cell2.col})`
          );
        }

        const result = backtrack(puzzle, state, config, stats, solutions);

        if (result) {
          // Found solution(s) and not finding all
          return true;
        }
      } else {
        stats.prunes++;
        if (config.debugLevel >= 2) {
          console.log(
            `Pruned domino ${domino.id} at (${edge.cell1.row},${edge.cell1.col})-(${edge.cell2.row},${edge.cell2.col}): constraint violation`
          );
        }
      }

      // Backtrack
      stats.backtracks++;
      state.gridPips[edge.cell1.row][edge.cell1.col] = null;
      state.gridPips[edge.cell2.row][edge.cell2.col] = null;
      state.usedDominoes = savedUsedDominoes;
      state.placements.pop();
      state.domains = savedDomains;
    }
  }

  return false;
}

/**
 * Check if the current state is a complete solution
 */
function isComplete(state: SolverState, puzzle: NormalizedPuzzle): boolean {
  for (let row = 0; row < puzzle.spec.rows; row++) {
    for (let col = 0; col < puzzle.spec.cols; col++) {
      // Holes are always "complete"
      if (puzzle.spec.regions[row]?.[col] === -1) {
        continue;
      }
      if (state.gridPips[row][col] === null) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Reconstruct state from solution (for explanation)
 */
function stateFromSolution(solution: Solution, puzzle: NormalizedPuzzle): SolverState {
  const used = new Map<string, number>();
  for (const d of solution.dominoes) {
    used.set(d.domino.id, (used.get(d.domino.id) || 0) + 1);
  }
  return {
    gridPips: solution.gridPips.map(row => [...row]),
    usedDominoes: used,
    placements: [...solution.dominoes],
    domains: new Map(),
  };
}

/**
 * Non-blocking solver that yields periodically
 * This is used by the UI to avoid freezing
 */
export async function solvePuzzleAsync(
  puzzle: NormalizedPuzzle,
  config: SolverConfig,
  onProgress?: (progress: SolverProgress) => void,
  signal?: { cancelled: boolean }
): Promise<SolverResult> {
  const startTime = Date.now();

  const initialState: SolverState = {
    gridPips: Array(puzzle.spec.rows)
      .fill(null)
      .map(() => Array(puzzle.spec.cols).fill(null)),
    usedDominoes: new Map(),
    placements: [],
    domains: initializeDomains(puzzle, config.maxPip),
  };

  const stats = {
    nodes: 0,
    backtracks: 0,
    prunes: 0,
    timeMs: 0,
  };

  const solutions: Solution[] = [];

  // Debug: Log initial state
  console.log('[SOLVER DEBUG] Initial domains:');
  for (const [key, domain] of initialState.domains) {
    console.log(`  ${key}: [${domain.join(',')}]`);
  }
  console.log('[SOLVER DEBUG] Edges count:', puzzle.edges.length);
  console.log('[SOLVER DEBUG] Domino tray:', JSON.stringify(puzzle.spec.dominoes));

  // Run backtracking with periodic yields
  await backtrackAsync(puzzle, initialState, config, stats, solutions, onProgress, signal);

  stats.timeMs = Date.now() - startTime;
  for (const sol of solutions) {
    sol.stats.timeMs = stats.timeMs;
    sol.stats.solutions = solutions.length;
  }

  if (solutions.length > 0) {
    return {
      success: true,
      solutions,
      explanation: explainSuccess(puzzle, stateFromSolution(solutions[0], puzzle)),
      stats,
    };
  } else {
    return {
      success: false,
      solutions: [],
      explanation: explainUnsatisfiable(puzzle, initialState),
      stats,
    };
  }
}

/**
 * Async backtracking with progress callbacks
 * Uses CELL-FIRST approach (like Python solver) to prevent orphaned cells
 */
async function backtrackAsync(
  puzzle: NormalizedPuzzle,
  state: SolverState,
  config: SolverConfig,
  stats: { nodes: number; backtracks: number; prunes: number },
  solutions: Solution[],
  onProgress?: (progress: SolverProgress) => void,
  signal?: { cancelled: boolean },
  depth: number = 0
): Promise<boolean> {
  // Check for cancellation
  if (signal?.cancelled) {
    return false;
  }

  // Yield to UI periodically
  if (stats.nodes % config.maxIterationsPerTick === 0) {
    await new Promise(resolve => setTimeout(resolve, 0));

    if (onProgress) {
      onProgress({
        nodes: stats.nodes,
        backtracks: stats.backtracks,
        prunes: stats.prunes,
        currentDepth: depth,
        completed: false,
        cancelled: signal?.cancelled || false,
      });
    }
  }

  stats.nodes++;

  if (isComplete(state, puzzle)) {
    if (isConsistent(puzzle, state)) {
      const solution: Solution = {
        gridPips: state.gridPips.map(row => [...row]),
        dominoes: [...state.placements],
        stats: {
          nodes: stats.nodes,
          backtracks: stats.backtracks,
          prunes: stats.prunes,
          timeMs: 0,
          solutions: solutions.length + 1,
        },
      };

      console.log('[SOLVER] *** SOLUTION FOUND! ***');
      solutions.push(solution);
      return !config.findAll;
    } else {
      console.log('[SOLVER] Complete but inconsistent!');
    }
  }

  // CELL-FIRST APPROACH: Pick a cell, then pair it with each unfilled neighbor
  const cell = selectNextCell(puzzle, state);
  if (!cell) {
    console.log('[SOLVER] No more unfilled cells');
    return false;
  }

  // Get unfilled neighbors to pair with
  const neighbors = getUnfilledNeighborsForCell(puzzle, state, cell);

  if (neighbors.length === 0) {
    // Cell has no unfilled neighbors - this is an orphan!
    stats.prunes++;
    if (depth < 8) {
      console.log(`[SOLVER] Cell (${cell.row},${cell.col}) is orphaned - no unfilled neighbors`);
    }
    return false;
  }

  if (depth === 0) {
    console.log(`[SOLVER DEBUG] First cell: (${cell.row},${cell.col})`);
    console.log(`[SOLVER DEBUG] Unfilled neighbors: ${neighbors.map(n => `(${n.row},${n.col})`).join(', ')}`);
  }

  // Try each neighbor
  for (const neighbor of neighbors) {
    if (signal?.cancelled) {
      return false;
    }

    // Create edge for this cell-neighbor pair
    const edge = { cell1: cell, cell2: neighbor };

    // Get candidate dominoes for this edge
    const candidates = getCandidateDominoes(
      puzzle,
      edge,
      state,
      config.maxPip,
      config.allowDuplicates
    );

    if (candidates.length === 0) {
      // No valid dominoes for this edge - try next neighbor
      continue;
    }

    if (depth === 0) {
      console.log(`[SOLVER DEBUG] First edge: (${cell.row},${cell.col})-(${neighbor.row},${neighbor.col})`);
      console.log(`[SOLVER DEBUG] Candidates: ${candidates.map(c => c.id).join(', ') || 'NONE'}`);
    }

    // Try each candidate domino
    for (const domino of candidates) {
      if (signal?.cancelled) {
        return false;
      }

      const placement: DominoPlacement = {
        domino,
        cell1: edge.cell1,
        cell2: edge.cell2,
      };

      const savedDomains = copyDomains(state.domains);
      const savedUsedDominoes = new Map(state.usedDominoes);

      state.gridPips[edge.cell1.row][edge.cell1.col] = domino.pip1;
      state.gridPips[edge.cell2.row][edge.cell2.col] = domino.pip2;
      state.usedDominoes.set(domino.id, (state.usedDominoes.get(domino.id) || 0) + 1);
      state.placements.push(placement);

      const prop1 = propagateConstraints(puzzle, state, edge.cell1, domino.pip1);
      const prop2 = prop1 && propagateConstraints(puzzle, state, edge.cell2, domino.pip2);

      if (depth < 8) {
        console.log(
          `[SOLVER] Depth ${depth}: Placed ${domino.id} at (${edge.cell1.row},${edge.cell1.col})-(${edge.cell2.row},${edge.cell2.col}) -> prop1=${prop1}, prop2=${prop2}`
        );
      }

      if (prop1 && prop2) {
        const result = await backtrackAsync(
          puzzle,
          state,
          config,
          stats,
          solutions,
          onProgress,
          signal,
          depth + 1
        );

        if (result) {
          return true;
        }
      } else {
        stats.prunes++;
      }

      stats.backtracks++;
      state.gridPips[edge.cell1.row][edge.cell1.col] = null;
      state.gridPips[edge.cell2.row][edge.cell2.col] = null;
      state.usedDominoes = savedUsedDominoes;
      state.placements.pop();
      state.domains = savedDomains;
    }
  }

  return false;
}

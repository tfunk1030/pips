import type { Puzzle, Solution, SpecReport } from '../model/types';
import { validateSpec } from '../validator/validateSpec';
import { pickNextCellMRV } from './heuristics';
import { regionFeasible, type RegionState } from './propagate';
import { explainRegionInfeasible } from './explain';

export type SolveProgress = {
  nodes: number;
  backtracks: number;
  prunes: number;
  elapsedMs: number;
  depth: number;
};

export type SolverStats = {
  nodes: number;
  backtracks: number;
  prunes: number;
  timeMs: number;
  yields: number;
};

export type SolveResult =
  | { kind: 'solved'; solution: Solution; stats: SolverStats }
  | { kind: 'unsat'; explanation: string; stats: SolverStats }
  | { kind: 'cancelled'; stats: SolverStats }
  | { kind: 'invalid_spec'; report: SpecReport };

export type SolveOptions = {
  mode: 'first' | 'all';
  yieldEvery: number;
  onProgress?: (p: SolveProgress) => void;
  signal?: AbortSignal;
  allowDuplicates?: boolean;
  maxPipOverride?: number;
  logLevel?: 'off' | 'info' | 'trace';
};

export type SolverState = {
  mate: Int16Array; // -1 or mate cellId
  pip: Int8Array; // -1 or 0..maxPip
  region: RegionState[];
  remainingDominoByType: number[]; // aligned with puzzle.dominoTypes
  unpaired: number;
};

type Move = {
  a: number;
  b: number;
  pipA: number;
  pipB: number;
  dominoTypeIndex: number;
};

type Undo = {
  a: number;
  b: number;
  oldMateA: number;
  oldMateB: number;
  oldPipA: number;
  oldPipB: number;
  touchedRegions: Array<{ ri: number; oldAssignedSum: number; oldUnassigned: number; oldAllEqVal: number }>;
  dominoTypeIndex: number;
};

export async function solvePuzzleAsync(puzzle: Puzzle, options: SolveOptions): Promise<SolveResult> {
  const start = Date.now();
  const report = validateSpec(puzzle);
  if (!report.ok) return { kind: 'invalid_spec', report };

  const p = applySolveOverrides(puzzle, options);

  const st: SolverState = {
    mate: new Int16Array(p.cells.length).fill(-1),
    pip: new Int8Array(p.cells.length).fill(-1),
    region: p.regionByIndex.map((r) => ({ assignedSum: 0, unassigned: r.cellIds.length, allEqVal: -1 })),
    remainingDominoByType: p.dominoTypes.map((d) => d.count),
    unpaired: p.cells.length,
  };

  const stats: SolverStats = { nodes: 0, backtracks: 0, prunes: 0, timeMs: 0, yields: 0 };
  let lastYieldNodes = 0;
  let lastProgressMs = 0;

  let firstUnsatExplanation: string | null = null;
  const solutions: Solution[] = [];

  const shouldYield = () => stats.nodes - lastYieldNodes >= Math.max(1, options.yieldEvery);
  const emitProgress = (depth: number) => {
    if (!options.onProgress) return;
    const now = Date.now();
    if (now - lastProgressMs < 80) return;
    lastProgressMs = now;
    options.onProgress({
      nodes: stats.nodes,
      backtracks: stats.backtracks,
      prunes: stats.prunes,
      elapsedMs: now - start,
      depth,
    });
  };

  const yieldToUI = async () => {
    if (!shouldYield()) return;
    lastYieldNodes = stats.nodes;
    stats.yields++;
    await new Promise<void>((r) => setTimeout(r, 0));
  };

  const checkAbort = () => {
    if (options.signal?.aborted) throw new AbortError();
  };

  const dfs = async (depth: number): Promise<void> => {
    checkAbort();
    if (st.unpaired === 0) {
      const sol = buildSolution(p, st);
      solutions.push(sol);
      return;
    }

    if (shouldYield()) {
      emitProgress(depth);
      await yieldToUI();
    }

    const cellId = pickNextCellMRV(p, st, (cid, best) => countMovesForCell(p, st, cid, best, stats));
    if (cellId < 0) return;

    const moves = enumerateMovesForCell(p, st, cellId, stats);
    if (moves.length === 0) {
      if (!firstUnsatExplanation) firstUnsatExplanation = explainDeadEnd(p, st, cellId);
      stats.backtracks++;
      return;
    }

    for (const mv of moves) {
      stats.nodes++;
      checkAbort();

      const undo = applyMove(p, st, mv, stats);
      if (!undo) {
        stats.prunes++;
        continue;
      }

      await dfs(depth + 1);

      undoMove(st, undo);

      if (options.mode === 'first' && solutions.length > 0) return;
    }

    stats.backtracks++;
  };

  try {
    await dfs(0);
  } catch (e) {
    if (e instanceof AbortError) {
      stats.timeMs = Date.now() - start;
      return { kind: 'cancelled', stats };
    }
    throw e;
  }

  stats.timeMs = Date.now() - start;
  if (solutions.length > 0) return { kind: 'solved', solution: solutions[0], stats };
  return { kind: 'unsat', explanation: firstUnsatExplanation ?? 'No solution exists.', stats };
}

function applySolveOverrides(p: Puzzle, options: SolveOptions): Puzzle {
  // If the domain looks like a default full set, allow overrides; otherwise keep the puzzle's explicit domain.
  const looksDefault =
    p.dominoTypes.length === ((p.maxPip + 1) * (p.maxPip + 2)) / 2 &&
    p.dominoTypes.every((d) => d.count === 1 || d.count > 10);

  if (!looksDefault) return p;

  const maxPip = options.maxPipOverride != null ? clampInt(options.maxPipOverride, 0, 12) : p.maxPip;
  const allowDuplicates = options.allowDuplicates != null ? !!options.allowDuplicates : p.allowDuplicates;
  if (maxPip === p.maxPip && allowDuplicates === p.allowDuplicates) return p;

  // Rebuild a default domain; keep everything else identical.
  const placements = p.cells.length / 2;
  const dominoTypes = [];
  for (let a = 0; a <= maxPip; a++) {
    for (let b = a; b <= maxPip; b++) {
      dominoTypes.push({ key: `${a},${b}`, a, b, count: allowDuplicates ? placements : 1 });
    }
  }
  return { ...p, maxPip, allowDuplicates, dominoTypes };
}

function enumerateMovesForCell(p: Puzzle, st: SolverState, cellId: number, stats: SolverStats): Move[] {
  const moves: Move[] = [];
  const neighs = p.neighbors[cellId];
  for (const b of neighs) {
    if (st.mate[b] !== -1) continue;
    for (let di = 0; di < p.dominoTypes.length; di++) {
      const d = p.dominoTypes[di];
      if (st.remainingDominoByType[di] <= 0) continue;

      // Orientation 1
      if (st.pip[cellId] === -1 && st.pip[b] === -1) {
        const mv1: Move = { a: cellId, b, pipA: d.a, pipB: d.b, dominoTypeIndex: di };
        if (quickFeasiblePair(p, st, mv1)) moves.push(mv1);
        else stats.prunes++;
        if (d.a !== d.b) {
          const mv2: Move = { a: cellId, b, pipA: d.b, pipB: d.a, dominoTypeIndex: di };
          if (quickFeasiblePair(p, st, mv2)) moves.push(mv2);
          else stats.prunes++;
        }
      }
    }
  }
  return moves;
}

function countMovesForCell(p: Puzzle, st: SolverState, cellId: number, bestSoFar: number, stats: SolverStats): number {
  let count = 0;
  const neighs = p.neighbors[cellId];
  for (const b of neighs) {
    if (st.mate[b] !== -1) continue;
    for (let di = 0; di < p.dominoTypes.length; di++) {
      if (st.remainingDominoByType[di] <= 0) continue;
      const d = p.dominoTypes[di];
      if (quickFeasiblePair(p, st, { a: cellId, b, pipA: d.a, pipB: d.b, dominoTypeIndex: di })) count++;
      else stats.prunes++;
      if (d.a !== d.b) {
        if (quickFeasiblePair(p, st, { a: cellId, b, pipA: d.b, pipB: d.a, dominoTypeIndex: di })) count++;
        else stats.prunes++;
      }
      if (count >= bestSoFar) return count;
    }
  }
  return count;
}

function quickFeasiblePair(p: Puzzle, st: SolverState, mv: Move): boolean {
  if (st.mate[mv.a] !== -1 || st.mate[mv.b] !== -1) return false;
  if (st.pip[mv.a] !== -1 || st.pip[mv.b] !== -1) return false;
  if (st.remainingDominoByType[mv.dominoTypeIndex] <= 0) return false;

  // Region feasibility after assigning both.
  const ra = p.regionIndexByCellId[mv.a];
  const rb = p.regionIndexByCellId[mv.b];

  // Copy only touched regions (at most 2) for the check.
  const tmpA = { ...st.region[ra] };
  const tmpB = ra === rb ? tmpA : { ...st.region[rb] };

  const okA = applyCellToRegion(p, tmpA, p.regionByIndex[ra].cellIds.length, p.regionByIndex[ra].constraint, mv.pipA);
  if (!okA) return false;
  const okB = applyCellToRegion(p, tmpB, p.regionByIndex[rb].cellIds.length, p.regionByIndex[rb].constraint, mv.pipB);
  if (!okB) return false;

  const fa = regionFeasible(p, p.regionByIndex[ra].cellIds.length, tmpA, p.regionByIndex[ra].constraint);
  if (!fa.ok) return false;
  const fb = regionFeasible(p, p.regionByIndex[rb].cellIds.length, tmpB, p.regionByIndex[rb].constraint);
  if (!fb.ok) return false;

  return true;
}

function applyMove(p: Puzzle, st: SolverState, mv: Move, stats: SolverStats): Undo | null {
  if (st.mate[mv.a] !== -1 || st.mate[mv.b] !== -1) return null;
  if (st.remainingDominoByType[mv.dominoTypeIndex] <= 0) return null;

  const touchedRegions: Undo['touchedRegions'] = [];
  const touch = (ri: number) => {
    if (touchedRegions.some((x) => x.ri === ri)) return;
    const rs = st.region[ri];
    touchedRegions.push({ ri, oldAssignedSum: rs.assignedSum, oldUnassigned: rs.unassigned, oldAllEqVal: rs.allEqVal });
  };

  const ra = p.regionIndexByCellId[mv.a];
  const rb = p.regionIndexByCellId[mv.b];
  touch(ra);
  touch(rb);

  const undo: Undo = {
    a: mv.a,
    b: mv.b,
    oldMateA: st.mate[mv.a],
    oldMateB: st.mate[mv.b],
    oldPipA: st.pip[mv.a],
    oldPipB: st.pip[mv.b],
    touchedRegions,
    dominoTypeIndex: mv.dominoTypeIndex,
  };

  // Assign mates
  st.mate[mv.a] = mv.b;
  st.mate[mv.b] = mv.a;
  st.unpaired -= 2;

  // Assign pips
  st.pip[mv.a] = mv.pipA;
  st.pip[mv.b] = mv.pipB;

  // Update domino remaining
  const cur = st.remainingDominoByType[mv.dominoTypeIndex];
  st.remainingDominoByType[mv.dominoTypeIndex] = Number.isFinite(cur) ? cur - 1 : cur;

  // Update regions
  const okA = applyCellToRegion(p, st.region[ra], p.regionByIndex[ra].cellIds.length, p.regionByIndex[ra].constraint, mv.pipA);
  const okB = applyCellToRegion(p, st.region[rb], p.regionByIndex[rb].cellIds.length, p.regionByIndex[rb].constraint, mv.pipB);
  if (!okA || !okB) {
    undoMove(st, undo);
    return null;
  }

  const fa = regionFeasible(p, p.regionByIndex[ra].cellIds.length, st.region[ra], p.regionByIndex[ra].constraint);
  if (!fa.ok) {
    stats.prunes++;
    undoMove(st, undo);
    return null;
  }
  const fb = regionFeasible(p, p.regionByIndex[rb].cellIds.length, st.region[rb], p.regionByIndex[rb].constraint);
  if (!fb.ok) {
    stats.prunes++;
    undoMove(st, undo);
    return null;
  }

  return undo;
}

function applyCellToRegion(p: Puzzle, rs: RegionState, regionSize: number, c: any, pipVal: number): boolean {
  // All assignments happen once per cell in this solver.
  rs.assignedSum += pipVal;
  rs.unassigned -= 1;

  if (c.all_equal) {
    if (rs.allEqVal === -1) rs.allEqVal = pipVal;
    else if (rs.allEqVal !== pipVal) return false;
  }

  // For all_equal regions with explicit sum equality, we can early-check divisibility when fully assigned, but
  // regionFeasible handles the general case.
  return rs.unassigned >= 0 && rs.assignedSum <= regionSize * p.maxPip;
}

function undoMove(st: SolverState, undo: Undo): void {
  st.mate[undo.a] = undo.oldMateA;
  st.mate[undo.b] = undo.oldMateB;

  st.pip[undo.a] = undo.oldPipA;
  st.pip[undo.b] = undo.oldPipB;

  st.unpaired += 2;

  // Restore domino count
  const cur = st.remainingDominoByType[undo.dominoTypeIndex];
  st.remainingDominoByType[undo.dominoTypeIndex] = Number.isFinite(cur) ? cur + 1 : cur;

  for (const t of undo.touchedRegions) {
    const rs = st.region[t.ri];
    rs.assignedSum = t.oldAssignedSum;
    rs.unassigned = t.oldUnassigned;
    rs.allEqVal = t.oldAllEqVal;
  }
}

function buildSolution(p: Puzzle, st: SolverState): Solution {
  const gridPips: number[][] = new Array(p.rows);
  for (let r = 0; r < p.rows; r++) {
    gridPips[r] = new Array(p.cols).fill(-1);
  }
  for (const cell of p.cells) {
    gridPips[cell.r][cell.c] = st.pip[cell.id];
  }

  const dominoes: Solution['dominoes'] = [];
  for (let a = 0; a < p.cells.length; a++) {
    const b = st.mate[a];
    if (b < a) continue;
    const ca = p.cells[a];
    const cb = p.cells[b];
    const pa = st.pip[a];
    const pb = st.pip[b];
    dominoes.push({
      cellA: { r: ca.r, c: ca.c },
      cellB: { r: cb.r, c: cb.c },
      pipsA: pa,
      pipsB: pb,
      dominoKey: `${Math.min(pa, pb)},${Math.max(pa, pb)}`,
    });
  }

  return {
    gridPips,
    mateCellIdByCellId: Array.from(st.mate),
    dominoes,
  };
}

class AbortError extends Error {
  constructor() {
    super('Solve cancelled');
    this.name = 'AbortError';
  }
}

function clampInt(n: number, lo: number, hi: number): number {
  const x = Math.trunc(Number(n));
  if (!Number.isFinite(x)) return lo;
  return Math.max(lo, Math.min(hi, x));
}

function explainDeadEnd(p: Puzzle, st: SolverState, cellId: number): string {
  const cell = p.cells[cellId];
  const ri = p.regionIndexByCellId[cellId];
  const region = p.regionByIndex[ri];
  const regionState = st.region[ri];

  const rf = regionFeasible(p, region.cellIds.length, regionState, region.constraint);
  if (!rf.ok) return explainRegionInfeasible(p, region.id, region.cellIds.length, regionState, region.constraint, rf.reason);

  const unpairedNeighbors = p.neighbors[cellId].filter((n) => st.mate[n] === -1);
  if (unpairedNeighbors.length === 0) {
    return `Cell (${cell.r},${cell.c}) is unpaired but has no unpaired neighbors (matching dead-end).`;
  }

  const anyDominoLeft = st.remainingDominoByType.some((c) => c > 0 || !Number.isFinite(c));
  if (!anyDominoLeft) return `No dominoes remaining but ${st.unpaired / 2} placements still needed.`;

  // Try to find a concrete failing region bound to report.
  for (const nb of unpairedNeighbors) {
    const rnb = p.regionIndexByCellId[nb];
    for (let di = 0; di < p.dominoTypes.length; di++) {
      if (st.remainingDominoByType[di] <= 0) continue;
      const d = p.dominoTypes[di];
      const tries: Array<[number, number]> = d.a === d.b ? [[d.a, d.b]] : [[d.a, d.b], [d.b, d.a]];
      for (const [pa, pb] of tries) {
        const tmpA = { ...st.region[ri] };
        const tmpB = ri === rnb ? tmpA : { ...st.region[rnb] };
        const okA = applyCellToRegion(p, tmpA, region.cellIds.length, region.constraint, pa);
        const okB = applyCellToRegion(p, tmpB, p.regionByIndex[rnb].cellIds.length, p.regionByIndex[rnb].constraint, pb);
        if (!okA || !okB) continue;
        const fa = regionFeasible(p, region.cellIds.length, tmpA, region.constraint);
        if (!fa.ok) return explainRegionInfeasible(p, region.id, region.cellIds.length, tmpA, region.constraint, fa.reason);
        const fb = regionFeasible(p, p.regionByIndex[rnb].cellIds.length, tmpB, p.regionByIndex[rnb].constraint);
        if (!fb.ok) {
          const regB = p.regionByIndex[rnb];
          return explainRegionInfeasible(p, regB.id, regB.cellIds.length, tmpB, regB.constraint, fb.reason);
        }
      }
    }
  }

  return `No valid domino placements for cell (${cell.r},${cell.c}). (All candidate pairings violate region constraints or domino availability.)`;
}



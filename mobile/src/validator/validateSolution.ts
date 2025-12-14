import type { Puzzle, Solution } from '../model/types';

export type ValidationError = { path: string; message: string };

export type RegionValidation = {
  regionId: string;
  cells: number;
  sum: number;
  constraint: string;
  ok: boolean;
  message?: string;
};

export type DominoUsage = {
  dominoKey: string;
  used: number;
  allowed: number | '∞';
  ok: boolean;
};

export type SolutionReport = {
  ok: boolean;
  errors: ValidationError[];
  regionReports: RegionValidation[];
  dominoUsage: DominoUsage[];
};

export function validateSolution(p: Puzzle, sol: Solution): SolutionReport {
  const errors: ValidationError[] = [];

  const numCells = p.cells.length;
  if (sol.mateCellIdByCellId.length !== numCells) {
    errors.push({ path: 'solution.mateCellIdByCellId', message: `Expected length ${numCells}.` });
    return { ok: false, errors, regionReports: [], dominoUsage: [] };
  }

  // Pip range + mate symmetry + adjacency + perfect matching.
  const seen = new Array<boolean>(numCells).fill(false);
  for (let a = 0; a < numCells; a++) {
    const b = sol.mateCellIdByCellId[a];
    if (b < 0 || b >= numCells) {
      errors.push({ path: `mate[${a}]`, message: `Invalid mate id ${b}.` });
      continue;
    }
    if (sol.mateCellIdByCellId[b] !== a) errors.push({ path: `mate[${a}]`, message: `Mate symmetry violated: mate[${a}]=${b} but mate[${b}]!=${a}.` });

    const ca = p.cells[a];
    const cb = p.cells[b];
    const manhattan = Math.abs(ca.r - cb.r) + Math.abs(ca.c - cb.c);
    if (manhattan !== 1) errors.push({ path: `mate[${a}]`, message: `Cells ${a} and ${b} not orthogonally adjacent.` });

    const pa = sol.gridPips[ca.r]?.[ca.c];
    const pb = sol.gridPips[cb.r]?.[cb.c];
    if (!Number.isInteger(pa) || pa < 0 || pa > p.maxPip) errors.push({ path: `gridPips[${ca.r}][${ca.c}]`, message: `Pip out of range: ${pa}` });
    if (!Number.isInteger(pb) || pb < 0 || pb > p.maxPip) errors.push({ path: `gridPips[${cb.r}][${cb.c}]`, message: `Pip out of range: ${pb}` });

    if (seen[a]) continue;
    if (seen[b]) errors.push({ path: 'matching', message: `Cell ${b} is paired more than once.` });
    seen[a] = true;
    seen[b] = true;
  }

  // Region checks.
  const regionReports: RegionValidation[] = [];
  for (const region of p.regionByIndex) {
    let sum = 0;
    let allEq: number | null = null;
    for (const cellId of region.cellIds) {
      const cell = p.cells[cellId];
      const v = sol.gridPips[cell.r]?.[cell.c];
      sum += v ?? 0;
      if (allEq == null) allEq = v ?? 0;
      else if ((v ?? 0) !== allEq) allEq = NaN;
    }

    const c = region.constraint;
    const parts: string[] = [];
    if (c.all_equal) parts.push('all_equal');
    const op = c.sum != null ? '=' : c.op;
    const val = c.sum != null ? c.sum : c.value;
    if (op && val != null) parts.push(`sum ${op} ${val}`);
    const constraint = parts.join(', ') || 'none';

    const { ok, message } = checkRegionConstraint(region.cellIds.length, sum, c, allEq);
    if (!ok && message) errors.push({ path: `region.${region.id}`, message });
    regionReports.push({ regionId: region.id, cells: region.cellIds.length, sum, constraint, ok, message });
  }

  // Domino multiset usage based on (a,b) keys.
  const usedCounts = new Map<string, number>();
  for (let a = 0; a < numCells; a++) {
    const b = sol.mateCellIdByCellId[a];
    if (b < a) continue;
    const ca = p.cells[a];
    const cb = p.cells[b];
    const pa = sol.gridPips[ca.r][ca.c];
    const pb = sol.gridPips[cb.r][cb.c];
    const key = `${Math.min(pa, pb)},${Math.max(pa, pb)}`;
    usedCounts.set(key, (usedCounts.get(key) ?? 0) + 1);
  }

  const allowedByKey = new Map(p.dominoTypes.map((d) => [d.key, d.count] as const));
  const dominoUsage: DominoUsage[] = [];
  const keys = Array.from(new Set([...usedCounts.keys(), ...allowedByKey.keys()])).sort();
  for (const key of keys) {
    const used = usedCounts.get(key) ?? 0;
    const allowed = allowedByKey.get(key);
    const allowedDisplay: number | '∞' = allowed == null ? 0 : Number.isFinite(allowed) ? allowed : '∞';
    const ok = allowed == null ? used === 0 : allowedDisplay === '∞' ? true : used <= allowedDisplay;
    if (!ok) errors.push({ path: `dominoes.${key}`, message: `Used ${used} of domino ${key} but only ${allowedDisplay} allowed.` });
    dominoUsage.push({ dominoKey: key, used, allowed: allowedDisplay, ok });
  }

  return { ok: errors.length === 0, errors, regionReports, dominoUsage };
}

function checkRegionConstraint(
  size: number,
  sum: number,
  c: { all_equal?: boolean; sum?: number; op?: any; value?: any },
  allEq: number | null
): { ok: boolean; message?: string } {
  if (c.all_equal) {
    if (allEq == null || Number.isNaN(allEq)) return { ok: false, message: `Region all_equal violated.` };
  }

  const op = c.sum != null ? '=' : (c.op as any);
  const val = c.sum != null ? c.sum : (c.value as any);
  if (op == null || val == null) return { ok: true };

  const v = Number(val);
  switch (op) {
    case '=':
      return sum === v ? { ok: true } : { ok: false, message: `Region sum expected ${v}, got ${sum}.` };
    case '<':
      return sum < v ? { ok: true } : { ok: false, message: `Region sum expected < ${v}, got ${sum}.` };
    case '>':
      return sum > v ? { ok: true } : { ok: false, message: `Region sum expected > ${v}, got ${sum}.` };
    case '≠':
      return sum !== v ? { ok: true } : { ok: false, message: `Region sum expected ≠ ${v}, got ${sum}.` };
    default:
      return { ok: false, message: `Unknown region op "${String(op)}".` };
  }
}




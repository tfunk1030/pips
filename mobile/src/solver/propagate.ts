import type { Puzzle, RegionConstraint } from '../model/types';

export type RegionState = {
  assignedSum: number;
  unassigned: number;
  allEqVal: number; // -1 if unset, else 0..maxPip
};

export function regionFeasible(p: Puzzle, regionSize: number, st: RegionState, c: RegionConstraint): { ok: boolean; reason?: string } {
  const maxPip = p.maxPip;

  // all_equal constraint
  if (c.all_equal) {
    if (st.allEqVal >= 0) {
      const total = st.allEqVal * regionSize;
      const sumOk = sumConstraintFeasible(total, total, c);
      if (!sumOk.ok) return sumOk;
    } else {
      // allEqVal unset: total must be k*v for some v in [0..maxPip]
      const minTotal = 0;
      const maxTotal = maxPip * regionSize;
      const sumOk = sumConstraintFeasible(minTotal, maxTotal, c, { step: regionSize });
      if (!sumOk.ok) return sumOk;
    }
    return { ok: true };
  }

  const minTotal = st.assignedSum + 0 * st.unassigned;
  const maxTotal = st.assignedSum + maxPip * st.unassigned;
  return sumConstraintFeasible(minTotal, maxTotal, c);
}

export function sumConstraintFeasible(
  minTotal: number,
  maxTotal: number,
  c: RegionConstraint,
  opts?: { step?: number }
): { ok: boolean; reason?: string } {
  const op = c.sum != null ? '=' : c.op;
  const val = c.sum != null ? c.sum : c.value;
  if (!op || val == null) return { ok: true };

  const target = val;

  // Optional step restriction (used for all_equal => total must be multiple of regionSize)
  const step = opts?.step;
  const rangeContainsWithStep = (x: number) => {
    if (x < minTotal || x > maxTotal) return false;
    if (!step) return true;
    return x % step === 0;
  };

  switch (op) {
    case '=':
      if (rangeContainsWithStep(target)) return { ok: true };
      return { ok: false, reason: `sum cannot reach ${target} (bounds [${minTotal},${maxTotal}])` };
    case '<':
      // Need some value < target
      if (minTotal < target) return { ok: true };
      return { ok: false, reason: `sum cannot be < ${target} (min=${minTotal})` };
    case '>':
      if (maxTotal > target) return { ok: true };
      return { ok: false, reason: `sum cannot be > ${target} (max=${maxTotal})` };
    case '≠':
      // Fails only if forced to equal target
      if (minTotal === maxTotal && minTotal === target && (!step || target % step === 0)) {
        return { ok: false, reason: `sum forced to ${target} but must be ≠` };
      }
      return { ok: true };
    default:
      return { ok: false, reason: `unknown op ${String(op)}` };
  }
}




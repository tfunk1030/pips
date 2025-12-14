import { NormalizedPuzzle } from '../model/normalize';
import { DomainMap } from './types';

export interface PropagationResult {
  domains: DomainMap;
  pruned: number;
  conflict?: string;
}

export function cloneDomains(domains: DomainMap): DomainMap {
  const next: DomainMap = {};
  Object.keys(domains).forEach((key) => {
    next[key] = [...domains[key]];
  });
  return next;
}

export function propagateRegionBounds(
  puzzle: NormalizedPuzzle,
  domains: DomainMap,
  assignment: Map<string, number>,
): PropagationResult {
  let pruned = 0;
  for (const [regionId, cells] of Object.entries(puzzle.regionCells)) {
    const constraint = puzzle.regionConstraints[Number(regionId)];
    if (!constraint) continue;
    const remainingCells: string[] = [];
    let assignedSum = 0;
    cells.forEach(([r, c]) => {
      const key = `${r},${c}`;
      if (assignment.has(key)) {
        assignedSum += assignment.get(key) ?? 0;
      } else {
        remainingCells.push(key);
      }
    });
    const maxDomainVal = puzzle.maxPip ?? 6;
    const minDomainVal = 0;
    const minPossible = assignedSum + remainingCells.length * minDomainVal;
    const maxPossible = assignedSum + remainingCells.length * maxDomainVal;

    if (constraint.type === 'sum') {
      if (constraint.value < minPossible || constraint.value > maxPossible) {
        return { domains, pruned, conflict: `Region ${regionId} cannot meet sum ${constraint.value}` };
      }
      // tighten domains roughly
      const slack = constraint.value - assignedSum;
      remainingCells.forEach((key) => {
        const domain = domains[key];
        const filtered = domain.filter((v) => v <= slack);
        pruned += domain.length - filtered.length;
        domains[key] = filtered;
        if (!domains[key].length) {
          return { domains, pruned, conflict: `Region ${regionId} exhausted for ${key}` };
        }
      });
    } else if (constraint.type === 'op') {
      if (constraint.op === '=' && (constraint.value < minPossible || constraint.value > maxPossible)) {
        return { domains, pruned, conflict: `Region ${regionId} cannot reach ${constraint.value}` };
      }
    }
  }
  return { domains, pruned };
}

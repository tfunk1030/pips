import { PuzzleSpec, SolutionGrid, ValidationIssue, ValidationReport } from '../model/types';

function unorderedId(a: number, b: number) {
  return a <= b ? `${a}-${b}` : `${b}-${a}`;
}

export function validateSolution(puzzle: PuzzleSpec, solution: SolutionGrid): ValidationReport {
  const issues: ValidationIssue[] = [];
  if (solution.gridPips.length !== puzzle.rows) {
    issues.push({ level: 'error', message: 'Solution rows mismatch' });
  }
  solution.gridPips.forEach((row, r) => {
    if (row.length !== puzzle.cols) {
      issues.push({ level: 'error', message: `Solution cols mismatch at row ${r}` });
    }
  });

  const seenCells: Record<string, boolean> = {};
  solution.dominoes.forEach((d) => {
    const [[r1, c1], [r2, c2]] = d.cells;
    const key1 = `${r1},${c1}`;
    const key2 = `${r2},${c2}`;
    if (seenCells[key1] || seenCells[key2]) {
      issues.push({ level: 'error', message: `Cell reused in domino ${d.id}` });
    }
    seenCells[key1] = true;
    seenCells[key2] = true;
    const dist = Math.abs(r1 - r2) + Math.abs(c1 - c2);
    if (dist !== 1) {
      issues.push({ level: 'error', message: `Domino ${d.id} not orthogonally adjacent` });
    }
  });

  const expectedCells = puzzle.rows * puzzle.cols;
  const covered = Object.keys(seenCells).length;
  if (covered !== expectedCells) {
    issues.push({ level: 'error', message: `Domino coverage mismatch (${covered}/${expectedCells})` });
  }

  // region constraints
  Object.entries(puzzle.regionConstraints).forEach(([id, constraint]) => {
    const cells: [number, number][] = [];
    puzzle.regions.forEach((row, r) =>
      row.forEach((regionId, c) => {
        if (regionId === Number(id)) cells.push([r, c]);
      }),
    );
    const values = cells.map(([r, c]) => solution.gridPips[r][c]);
    if (constraint.type === 'sum') {
      const s = values.reduce((a, b) => a + b, 0);
      if (s !== constraint.value) {
        issues.push({ level: 'error', message: `Region ${id} sum ${s} != ${constraint.value}` });
      }
    } else if (constraint.type === 'op') {
      const s = values.reduce((a, b) => a + b, 0);
      if (constraint.op === '=' && s !== constraint.value) {
        issues.push({ level: 'error', message: `Region ${id} must equal ${constraint.value}` });
      }
      if (constraint.op === '<' && !(s < constraint.value)) issues.push({ level: 'error', message: `Region ${id} < ${constraint.value}` });
      if (constraint.op === '>' && !(s > constraint.value)) issues.push({ level: 'error', message: `Region ${id} > ${constraint.value}` });
      if (constraint.op === '≠' && s === constraint.value) issues.push({ level: 'error', message: `Region ${id} ≠ ${constraint.value}` });
    } else if (constraint.type === 'all_equal') {
      const unique = new Set(values);
      if (unique.size !== 1) {
        issues.push({ level: 'error', message: `Region ${id} not all equal` });
      }
    }
  });

  // domino uniqueness
  const seenDomino: Record<string, number> = {};
  solution.dominoes.forEach((d) => {
    const pair = unorderedId(d.values[0], d.values[1]);
    seenDomino[pair] = (seenDomino[pair] ?? 0) + 1;
  });
  if (!puzzle.allowDuplicates) {
    Object.entries(seenDomino).forEach(([pair, count]) => {
      if (count > 1) {
        issues.push({ level: 'error', message: `Domino ${pair} reused ${count} times` });
      }
    });
  }

  return { ok: !issues.some((i) => i.level === 'error'), issues };
}

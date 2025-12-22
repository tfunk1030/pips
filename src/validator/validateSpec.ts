import { PuzzleSpec, ValidationIssue, ValidationReport } from '../model/types';

export function validateSpec(puzzle: PuzzleSpec): ValidationReport {
  const issues: ValidationIssue[] = [];
  if (puzzle.rows <= 0 || puzzle.cols <= 0) {
    issues.push({ level: 'error', message: 'Rows and cols must be positive' });
  }
  if (puzzle.regions.length !== puzzle.rows) {
    issues.push({ level: 'error', message: 'Regions row count mismatch' });
  }
  puzzle.regions.forEach((row, r) => {
    if (row.length !== puzzle.cols) {
      issues.push({ level: 'error', message: `Region width mismatch at row ${r}` });
    }
  });
  const seen: Record<number, boolean> = {};
  puzzle.regions.flat().forEach((id) => {
    seen[id] = true;
  });
  Object.keys(puzzle.regionConstraints).forEach((id) => {
    if (!seen[Number(id)]) {
      issues.push({ level: 'warning', message: `Constraint for unused region ${id}` });
    }
  });
  Object.entries(puzzle.regionConstraints).forEach(([id, c]) => {
    if (c.type === 'sum' && typeof c.value !== 'number') {
      issues.push({ level: 'error', message: `Region ${id} sum missing value` });
    }
  });
  return { ok: !issues.some((i) => i.level === 'error'), issues };
}

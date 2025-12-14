import type { Puzzle, SpecReport } from '../model/types';

export function validateSpec(p: Puzzle): SpecReport {
  const errors: SpecReport['errors'] = [];
  const warnings: SpecReport['warnings'] = [];

  if (p.rows <= 0 || p.cols <= 0) errors.push({ path: 'grid', message: 'rows/cols must be > 0.' });
  if (p.cells.length % 2 !== 0) errors.push({ path: 'grid', message: 'Number of cells must be even.' });
  if (p.maxPip < 0) errors.push({ path: 'dominoes.maxPip', message: 'maxPip must be >= 0.' });

  // Regions cover all cells.
  for (const cell of p.cells) {
    if (cell.regionId == null || cell.regionId === '') errors.push({ path: 'regions', message: `Cell (${cell.r},${cell.c}) missing region.` });
  }

  // Region constraints basic sanity
  for (const region of p.regionByIndex) {
    const c = region.constraint;
    if (c.size != null && c.size !== region.cellIds.length) {
      errors.push({
        path: `regionConstraints.${region.id}.size`,
        message: `Expected size ${c.size} but region has ${region.cellIds.length}.`,
      });
    }
    if (c.sum != null && c.sum < 0) errors.push({ path: `regionConstraints.${region.id}.sum`, message: 'sum must be >= 0.' });
    if (c.op != null && c.value == null) errors.push({ path: `regionConstraints.${region.id}`, message: 'op requires value.' });
  }

  // Domino feasibility: at least enough capacity if unique.
  if (!p.allowDuplicates) {
    const need = p.cells.length / 2;
    const have = p.dominoTypes.reduce((acc, d) => acc + (Number.isFinite(d.count) ? d.count : 0), 0);
    if (have < need) {
      errors.push({
        path: 'dominoes',
        message: `Not enough dominoes for grid: need ${need}, have ${have}.`,
      });
    }
  } else {
    // If duplicates allowed, counts are typically large/infinite; warn if domain is empty.
    if (p.dominoTypes.length === 0) errors.push({ path: 'dominoes', message: 'No domino domain.' });
  }

  return { ok: errors.length === 0, errors, warnings };
}




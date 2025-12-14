import type {
  DominoType,
  Op,
  Puzzle,
  RawPuzzle,
  RawPuzzleFormalV1,
  RawPuzzleLegacyAsciiV0,
  RegionConstraint,
} from './types';

export function normalizePuzzle(raw: RawPuzzle): Puzzle {
  if (isLegacy(raw)) return normalizeLegacy(raw);
  return normalizeFormal(raw);
}

function isLegacy(raw: RawPuzzle): raw is RawPuzzleLegacyAsciiV0 {
  return (raw as any).board != null;
}

function normalizeFormal(raw: RawPuzzleFormalV1): Puzzle {
  const rows = mustInt(raw.grid.rows, 'grid.rows');
  const cols = mustInt(raw.grid.cols, 'grid.cols');
  if (rows <= 0 || cols <= 0) throw new Error('grid.rows and grid.cols must be > 0.');

  const cellsMask = parseAsciiMask(raw.grid.cells ?? defaultAsciiMask(rows, cols), rows, cols, 'grid.cells');
  const regionGrid = parseRegions(raw.regions, rows, cols, cellsMask);
  const regionConstraints = raw.regionConstraints ?? {};

  const maxPip = clampInt(raw.dominoes?.maxPip ?? 6, 0, 12);
  const allowDuplicates = !(raw.dominoes?.unique ?? true);
  const dominoTypes = buildDominoTypes(maxPip, allowDuplicates, raw.dominoes?.tiles, rows, cols, cellsMask);

  return buildPuzzle(rows, cols, cellsMask, regionGrid, regionConstraints, maxPip, allowDuplicates, dominoTypes);
}

function normalizeLegacy(raw: RawPuzzleLegacyAsciiV0): Puzzle {
  const shapeLines = toLines(raw.board.shape, 'board.shape');
  const regionLines = toLines(raw.board.regions, 'board.regions');
  if (shapeLines.length !== regionLines.length) throw new Error('board.shape and board.regions must have same row count.');
  const rows = shapeLines.length;
  const cols = shapeLines[0]?.length ?? 0;
  if (rows === 0 || cols === 0) throw new Error('board.shape must be non-empty.');
  for (let r = 0; r < rows; r++) {
    if (shapeLines[r].length !== cols) throw new Error('board.shape rows must have consistent width.');
    if (regionLines[r].length !== cols) throw new Error('board.regions rows must have consistent width.');
  }

  const cellsMask = parseAsciiMask(shapeLines, rows, cols, 'board.shape');
  const regionGrid = parseRegions(regionLines, rows, cols, cellsMask);

  const maxPip = clampInt(raw.pips?.pip_max ?? 6, 0, 12);
  const allowDuplicates = !(raw.dominoes?.unique ?? true);

  const legacyConstraints: Record<string, RegionConstraint> = {};
  const rc = raw.region_constraints ?? {};
  for (const [rid, c] of Object.entries(rc)) {
    if (c.type === 'all_equal') legacyConstraints[rid] = { all_equal: true };
    else {
      const op = legacyOpToOp(c.op);
      legacyConstraints[rid] = { op, value: mustInt(c.value, `region_constraints.${rid}.value`) };
    }
  }

  const dominoTypes = buildDominoTypes(maxPip, allowDuplicates, raw.dominoes?.tiles, rows, cols, cellsMask);
  return buildPuzzle(rows, cols, cellsMask, regionGrid, legacyConstraints, maxPip, allowDuplicates, dominoTypes);
}

function buildPuzzle(
  rows: number,
  cols: number,
  cellsMask: boolean[],
  regionGrid: (string | null)[],
  regionConstraints: Record<string, RegionConstraint>,
  maxPip: number,
  allowDuplicates: boolean,
  dominoTypes: DominoType[]
): Puzzle {
  const rcToCellId = new Array(rows * cols).fill(-1);
  const cells: Puzzle['cells'] = [];
  let id = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const idx = r * cols + c;
      if (!cellsMask[idx]) continue;
      const regionId = regionGrid[idx];
      if (!regionId) throw new Error(`Missing region for cell (${r},${c}).`);
      rcToCellId[idx] = id;
      cells.push({ id, r, c, regionId: String(regionId) });
      id++;
    }
  }

  if (cells.length % 2 !== 0) {
    throw new Error(`Grid has ${cells.length} cells, but must be even to tile with dominoes.`);
  }

  const regionIds = Array.from(new Set(cells.map((x) => x.regionId))).sort(stableCompare);
  const regionIndexById: Record<string, number> = Object.create(null);
  regionIds.forEach((rid, i) => (regionIndexById[rid] = i));

  const regionByIndex = regionIds.map((rid) => ({
    id: rid,
    cellIds: [] as number[],
    constraint: normalizeConstraint(regionConstraints[rid] ?? {}),
  }));

  const regionIndexByCellId = new Array(cells.length).fill(-1);
  for (const cell of cells) {
    const ri = regionIndexById[cell.regionId];
    regionIndexByCellId[cell.id] = ri;
    regionByIndex[ri].cellIds.push(cell.id);
  }

  // Neighbors (orthogonal), deterministic order: right, down, left, up.
  const neighbors: number[][] = new Array(cells.length);
  for (const cell of cells) {
    const n: number[] = [];
    const tryAdd = (rr: number, cc: number) => {
      if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) return;
      const nid = rcToCellId[rr * cols + cc];
      if (nid >= 0) n.push(nid);
    };
    tryAdd(cell.r, cell.c + 1);
    tryAdd(cell.r + 1, cell.c);
    tryAdd(cell.r, cell.c - 1);
    tryAdd(cell.r - 1, cell.c);
    neighbors[cell.id] = n;
  }

  // Apply region size sanity checks (hard error).
  for (const region of regionByIndex) {
    const expected = region.constraint.size;
    if (expected != null && expected !== region.cellIds.length) {
      throw new Error(`Region ${region.id} expected size ${expected} but has ${region.cellIds.length} cells.`);
    }
  }

  return {
    rows,
    cols,
    maxPip,
    allowDuplicates,
    rcToCellId,
    cells,
    neighbors,
    regionIds,
    regionIndexById,
    regionByIndex,
    regionIndexByCellId,
    dominoTypes,
  };
}

function normalizeConstraint(c: RegionConstraint): RegionConstraint {
  const out: RegionConstraint = { ...c };
  if (out.sum != null) {
    out.op = '=';
    out.value = out.sum;
  }
  if (out.op != null && out.value == null) throw new Error(`Region constraint has op=${out.op} but no value.`);
  return out;
}

function buildDominoTypes(
  maxPip: number,
  allowDuplicates: boolean,
  tiles: Array<[number, number]> | undefined,
  rows: number,
  cols: number,
  cellsMask: boolean[]
): DominoType[] {
  const cellCount = cellsMask.reduce((acc, x) => acc + (x ? 1 : 0), 0);
  const placements = Math.floor(cellCount / 2);

  const counts = new Map<string, { a: number; b: number; count: number }>();
  const add = (a0: number, b0: number, count: number) => {
    const a = Math.min(a0, b0);
    const b = Math.max(a0, b0);
    if (a < 0 || b > maxPip) throw new Error(`Domino [${a0},${b0}] out of range for maxPip=${maxPip}.`);
    const key = `${a},${b}`;
    const cur = counts.get(key);
    if (!cur) counts.set(key, { a, b, count });
    else cur.count = cur.count + count;
  };

  if (tiles && tiles.length > 0) {
    for (const [a, b] of tiles) add(mustInt(a, 'dominoes.tiles[][0]'), mustInt(b, 'dominoes.tiles[][1]'), 1);
  } else {
    // Default: full double-N set.
    for (let a = 0; a <= maxPip; a++) for (let b = a; b <= maxPip; b++) add(a, b, allowDuplicates ? placements : 1);
  }

  const out = Array.from(counts.values())
    .map((x) => ({ key: `${x.a},${x.b}`, a: x.a, b: x.b, count: x.count }))
    .sort((p, q) => p.a - q.a || p.b - q.b);

  if (!allowDuplicates) {
    const total = out.reduce((acc, d) => acc + d.count, 0);
    if (total < placements) {
      throw new Error(`Not enough dominoes (${total}) to fill ${placements} placements (grid has ${cellCount} cells).`);
    }
  }

  return out;
}

function parseRegions(
  regions: RawPuzzleFormalV1['regions'],
  rows: number,
  cols: number,
  cellsMask: boolean[]
): (string | null)[] {
  const out: (string | null)[] = new Array(rows * cols).fill(null);

  // Mapping: { "r,c": "A" }
  if (regions && typeof regions === 'object' && !Array.isArray(regions)) {
    for (const [k, v] of Object.entries(regions)) {
      const m = /^(\d+)\s*,\s*(\d+)$/.exec(k);
      if (!m) throw new Error(`Invalid regions key "${k}". Expected "r,c".`);
      const r = Number(m[1]);
      const c = Number(m[2]);
      if (r < 0 || r >= rows || c < 0 || c >= cols) throw new Error(`regions["${k}"] out of bounds.`);
      const idx = r * cols + c;
      if (!cellsMask[idx]) continue;
      out[idx] = String(v);
    }
    return out;
  }

  // 2D arrays
  if (Array.isArray(regions)) {
    if (regions.length !== rows) throw new Error(`regions must have ${rows} rows.`);
    for (let r = 0; r < rows; r++) {
      const row = regions[r] as any;
      if (Array.isArray(row)) {
        if (row.length !== cols) throw new Error(`regions row ${r} must have ${cols} cols.`);
        for (let c = 0; c < cols; c++) {
          const idx = r * cols + c;
          if (!cellsMask[idx]) continue;
          out[idx] = String(row[c]);
        }
      } else if (typeof row === 'string') {
        if (row.length !== cols) throw new Error(`regions row ${r} must have width ${cols}.`);
        for (let c = 0; c < cols; c++) {
          const idx = r * cols + c;
          const ch = row[c];
          if (!cellsMask[idx]) continue;
          if (ch === '#') continue;
          out[idx] = ch;
        }
      } else {
        throw new Error('regions rows must be arrays or strings.');
      }
    }
    return out;
  }

  // ASCII string
  if (typeof regions === 'string') {
    const lines = toLines(regions, 'regions');
    if (lines.length !== rows) throw new Error(`regions must have ${rows} lines.`);
    for (let r = 0; r < rows; r++) {
      if (lines[r].length !== cols) throw new Error(`regions line ${r} must have width ${cols}.`);
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        if (!cellsMask[idx]) continue;
        const ch = lines[r][c];
        if (ch === '#') continue;
        out[idx] = ch;
      }
    }
    return out;
  }

  throw new Error('Unsupported regions format.');
}

function parseAsciiMask(mask: string | string[], rows: number, cols: number, path: string): boolean[] {
  const lines = toLines(mask, path);
  if (lines.length !== rows) throw new Error(`${path} must have ${rows} lines.`);
  for (let r = 0; r < rows; r++) {
    if (lines[r].length !== cols) throw new Error(`${path} line ${r} must have width ${cols}.`);
  }
  const out: boolean[] = new Array(rows * cols).fill(false);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const ch = lines[r][c];
      out[r * cols + c] = ch !== '#';
    }
  }
  return out;
}

function defaultAsciiMask(rows: number, cols: number): string[] {
  const line = '.'.repeat(cols);
  return new Array(rows).fill(line);
}

function toLines(x: string | string[], path: string): string[] {
  const lines = Array.isArray(x) ? x : String(x).split(/\r?\n/);
  const out = lines.map((s) => s.replace(/\s+$/g, '')).filter((s) => s.length > 0);
  if (out.length === 0) throw new Error(`${path} must not be empty.`);
  return out;
}

function legacyOpToOp(op?: '==' | '!=' | '<' | '>'): Op {
  switch (op) {
    case '==':
      return '=';
    case '!=':
      return '≠';
    case '<':
      return '<';
    case '>':
      return '>';
    default:
      return '=';
  }
}

function mustInt(x: any, path: string): number {
  const n = typeof x === 'number' ? x : Number.parseInt(String(x), 10);
  if (!Number.isFinite(n)) throw new Error(`Expected number at ${path}.`);
  return Math.trunc(n);
}

function clampInt(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.trunc(n)));
}

function stableCompare(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}




import YAML from 'yaml';
import { PuzzleSpec, RegionConstraint } from './types';

export interface ParsedPuzzle {
  puzzle: PuzzleSpec;
  raw: any;
}

function normalizeConstraint(input: any): RegionConstraint {
  if (!input) {
    throw new Error('Missing region constraint');
  }
  if (input.sum !== undefined) {
    return { type: 'sum', value: Number(input.sum), size: input.size };
  }
  if (input.op && input.value !== undefined) {
    return { type: 'op', op: input.op, value: Number(input.value), size: input.size } as RegionConstraint;
  }
  if (input.all_equal) {
    return { type: 'all_equal', size: input.size };
  }
  throw new Error(`Unknown constraint: ${JSON.stringify(input)}`);
}

export function parsePuzzle(text: string): ParsedPuzzle {
  const raw = YAML.parse(text);
  if (!raw || typeof raw !== 'object') {
    throw new Error('Invalid puzzle payload');
  }
  const rows = Number(raw.rows);
  const cols = Number(raw.cols);
  if (!rows || !cols) {
    throw new Error('Puzzle must define rows and cols');
  }
  if (!Array.isArray(raw.regions) || raw.regions.length !== rows) {
    throw new Error('regions must be a rows-length array');
  }
  const regions = raw.regions.map((row: any) => {
    if (!Array.isArray(row) || row.length !== cols) {
      throw new Error('Each regions row must match column length');
    }
    return row.map(Number);
  });

  const constraints: Record<number, RegionConstraint> = {};
  if (!raw.regionConstraints || typeof raw.regionConstraints !== 'object') {
    throw new Error('regionConstraints required');
  }
  Object.entries(raw.regionConstraints).forEach(([key, value]) => {
    const id = Number(key);
    constraints[id] = normalizeConstraint(value);
  });

  const puzzle: PuzzleSpec = {
    name: raw.name,
    rows,
    cols,
    regions,
    regionConstraints: constraints,
    maxPip: raw.maxPip ?? 6,
    allowDuplicates: Boolean(raw.allowDuplicates),
  };
  return { puzzle, raw };
}

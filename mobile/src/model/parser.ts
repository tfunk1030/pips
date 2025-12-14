import YAML from 'js-yaml';
import type { RawPuzzle, RawPuzzleFormalV1, RawPuzzleLegacyAsciiV0 } from './types';

export function parsePuzzleText(text: string): RawPuzzle {
  const trimmed = text.trim();
  if (!trimmed) throw new Error('Empty puzzle text.');

  // Try JSON first (nice for debugging).
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const obj = JSON.parse(trimmed);
      return coerceRawPuzzle(obj);
    } catch {
      // fall through to YAML
    }
  }

  const doc = YAML.load(trimmed);
  return coerceRawPuzzle(doc);
}

function coerceRawPuzzle(obj: any): RawPuzzle {
  if (!obj || typeof obj !== 'object') throw new Error('Puzzle must be a YAML/JSON object.');

  // Legacy schema detection
  if (obj.board && obj.board.shape && obj.board.regions) {
    return obj as RawPuzzleLegacyAsciiV0;
  }

  // Formal schema detection
  if (obj.grid && typeof obj.grid.rows === 'number' && typeof obj.grid.cols === 'number' && obj.regions) {
    return obj as RawPuzzleFormalV1;
  }

  throw new Error(
    'Unrecognized puzzle schema. Expected either { grid: {rows, cols}, regions: ... } or legacy { board: {shape, regions}, ... }.'
  );
}




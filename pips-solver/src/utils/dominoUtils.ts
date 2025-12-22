/**
 * Domino utility functions
 * Parsing, validation, and display helpers
 */

import { DominoPair } from '../model/overlayTypes';

/**
 * Parse domino shorthand input
 * Accepts formats like:
 * - "61 33 36 43" (space-separated pairs)
 * - "6-1 3-3 3-6" (dash-separated)
 * - "6,1 3,3 3,6" (comma-separated)
 * - "61336343" (continuous digits)
 *
 * Extracts pairs of digits 0-6
 */
export function parseDominoShorthand(input: string): DominoPair[] {
  // Remove all non-digit characters, keep only 0-6
  const digits = input.replace(/[^0-6]/g, '');

  const pairs: DominoPair[] = [];
  for (let i = 0; i < digits.length - 1; i += 2) {
    pairs.push([parseInt(digits[i], 10), parseInt(digits[i + 1], 10)]);
  }

  return pairs;
}

/**
 * Convert dominoes to shorthand string
 */
export function dominoesToShorthand(dominoes: DominoPair[]): string {
  return dominoes.map(([a, b]) => `${a}${b}`).join(' ');
}

/**
 * Calculate required domino count based on cell count
 */
export function calculateRequiredDominoes(cellCount: number): number {
  return Math.floor(cellCount / 2);
}

/**
 * Validate domino count against cell count
 */
export function validateDominoCount(
  dominoes: DominoPair[],
  cellCount: number
): { valid: boolean; message: string } {
  const required = calculateRequiredDominoes(cellCount);
  const actual = dominoes.length;

  if (actual === required) {
    return {
      valid: true,
      message: `✓ ${actual} dominoes for ${cellCount} cells`,
    };
  }

  if (actual < required) {
    return {
      valid: false,
      message: `Need ${required - actual} more dominoes (${actual}/${required})`,
    };
  }

  return {
    valid: false,
    message: `Too many dominoes: have ${actual}, need ${required}`,
  };
}

/**
 * Generate all unique domino pairs for a given max pip value
 */
export function generateAllDominoes(maxPip: number = 6): DominoPair[] {
  const dominoes: DominoPair[] = [];
  for (let i = 0; i <= maxPip; i++) {
    for (let j = i; j <= maxPip; j++) {
      dominoes.push([i, j]);
    }
  }
  return dominoes;
}

/**
 * Check if two dominoes are the same (order-independent)
 */
export function dominoesEqual(a: DominoPair, b: DominoPair): boolean {
  return (a[0] === b[0] && a[1] === b[1]) || (a[0] === b[1] && a[1] === b[0]);
}

/**
 * Find duplicate dominoes in a list
 */
export function findDuplicateDominoes(dominoes: DominoPair[]): number[] {
  const seen: DominoPair[] = [];
  const duplicateIndices: number[] = [];

  for (let i = 0; i < dominoes.length; i++) {
    const domino = dominoes[i];
    const isDuplicate = seen.some(s => dominoesEqual(s, domino));

    if (isDuplicate) {
      duplicateIndices.push(i);
    } else {
      seen.push(domino);
    }
  }

  return duplicateIndices;
}

/**
 * Get pip display positions for a given value (0-6)
 * Returns array of [x, y] positions in 0-1 range
 */
export function getPipPositions(value: number): [number, number][] {
  const m = 0.25; // margin
  const c = 0.5;  // center

  const positions: { [key: number]: [number, number][] } = {
    0: [],
    1: [[c, c]],
    2: [[m, m], [1 - m, 1 - m]],
    3: [[m, m], [c, c], [1 - m, 1 - m]],
    4: [[m, m], [1 - m, m], [m, 1 - m], [1 - m, 1 - m]],
    5: [[m, m], [1 - m, m], [c, c], [m, 1 - m], [1 - m, 1 - m]],
    6: [[m, m], [1 - m, m], [m, c], [1 - m, c], [m, 1 - m], [1 - m, 1 - m]],
  };

  return positions[value] || [];
}

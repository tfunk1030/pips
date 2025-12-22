/**
 * Domino Validator
 *
 * Validates domino extraction results.
 */

import {
  DominoExtractionResult,
  CellDetectionResult,
  ValidationResult,
  ValidationError,
} from '../types';
import { NYT_VALIDATION } from '../config';

/**
 * Validate domino extraction result
 */
export function validateDominoes(
  result: DominoExtractionResult,
  cells: CellDetectionResult
): ValidationResult {
  const errors: ValidationError[] = [];

  // Calculate expected count
  const cellCount = (cells.shape.match(/\./g) || []).length;
  const expectedCount = cellCount / 2;

  // Check count
  if (result.dominoes.length !== expectedCount) {
    errors.push({
      stage: 'dominoes',
      field: 'dominoes',
      message: `Found ${result.dominoes.length} dominoes, expected ${expectedCount} (for ${cellCount} cells)`,
      value: { found: result.dominoes.length, expected: expectedCount },
    });
  }

  // Check pip ranges
  const [minPip, maxPip] = NYT_VALIDATION.pipRange;

  for (let i = 0; i < result.dominoes.length; i++) {
    const [p1, p2] = result.dominoes[i];

    if (p1 < minPip || p1 > maxPip) {
      errors.push({
        stage: 'dominoes',
        field: 'dominoes',
        message: `Domino ${i + 1}: pip value ${p1} outside range [${minPip}, ${maxPip}]`,
        value: p1,
      });
    }

    if (p2 < minPip || p2 > maxPip) {
      errors.push({
        stage: 'dominoes',
        field: 'dominoes',
        message: `Domino ${i + 1}: pip value ${p2} outside range [${minPip}, ${maxPip}]`,
        value: p2,
      });
    }
  }

  // Check for duplicates (if uniqueDominoes is enabled)
  if (NYT_VALIDATION.uniqueDominoes) {
    const seen = new Map<string, number>();

    for (let i = 0; i < result.dominoes.length; i++) {
      const [p1, p2] = result.dominoes[i];
      // Normalize: smaller pip first
      const key = `${Math.min(p1, p2)}-${Math.max(p1, p2)}`;

      if (seen.has(key)) {
        errors.push({
          stage: 'dominoes',
          field: 'dominoes',
          message: `Duplicate domino [${p1}, ${p2}] at positions ${seen.get(key)! + 1} and ${i + 1}`,
          value: { domino: [p1, p2], positions: [seen.get(key)! + 1, i + 1] },
        });
      } else {
        seen.set(key, i);
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Check if dominoes can cover the grid given constraints
 * This is a quick feasibility check, not a full solve
 */
export function checkDominoFeasibility(
  dominoes: [number, number][],
  cellCount: number,
  constraints: Record<string, { type: string; op?: string; value?: number }>
): { feasible: boolean; issues: string[] } {
  const issues: string[] = [];

  // Check total pip coverage
  const totalPips = dominoes.reduce((sum, [p1, p2]) => sum + p1 + p2, 0);

  // For sum constraints, do a quick sanity check
  let totalSumTargets = 0;
  let hasAllEqual = false;

  for (const constraint of Object.values(constraints)) {
    if (constraint.type === 'sum' && constraint.op === '==' && constraint.value !== undefined) {
      totalSumTargets += constraint.value;
    }
    if (constraint.type === 'all_equal') {
      hasAllEqual = true;
    }
  }

  // If we have sum constraints covering all cells, total should match
  // This is a heuristic - not all cells may be covered by sum constraints
  if (totalSumTargets > 0 && totalSumTargets > totalPips) {
    issues.push(
      `Sum constraints total ${totalSumTargets} but domino pips total only ${totalPips}`
    );
  }

  // Check if we have enough variety for all_equal constraints
  if (hasAllEqual) {
    const pipCounts = new Map<number, number>();
    for (const [p1, p2] of dominoes) {
      pipCounts.set(p1, (pipCounts.get(p1) || 0) + 1);
      pipCounts.set(p2, (pipCounts.get(p2) || 0) + 1);
    }

    // all_equal needs at least 2 of the same pip value
    let hasRepeats = false;
    for (const count of pipCounts.values()) {
      if (count >= 2) {
        hasRepeats = true;
        break;
      }
    }

    if (!hasRepeats) {
      issues.push('all_equal constraint present but no repeated pip values in dominoes');
    }
  }

  return {
    feasible: issues.length === 0,
    issues,
  };
}

/**
 * Get statistics about the domino set
 */
export function getDominoStats(dominoes: [number, number][]): {
  count: number;
  totalPips: number;
  pipDistribution: Record<number, number>;
  hasDoubles: boolean;
  doubles: number[];
} {
  const pipDistribution: Record<number, number> = {};
  let totalPips = 0;
  const doubles: number[] = [];

  for (const [p1, p2] of dominoes) {
    totalPips += p1 + p2;
    pipDistribution[p1] = (pipDistribution[p1] || 0) + 1;
    pipDistribution[p2] = (pipDistribution[p2] || 0) + 1;

    if (p1 === p2) {
      doubles.push(p1);
    }
  }

  return {
    count: dominoes.length,
    totalPips,
    pipDistribution,
    hasDoubles: doubles.length > 0,
    doubles,
  };
}

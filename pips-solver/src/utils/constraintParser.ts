/**
 * Constraint shorthand parser
 * Parses strings like "A=8 B>4 C= Dx" into constraint definitions
 */

import { ConstraintDef, ConstraintOp, ConstraintType } from '../model/overlayTypes';

/**
 * Parse constraint shorthand into a map of region constraints
 *
 * Format examples:
 * - "A=8" or "A8" → Region A: sum equals 8
 * - "B>4" → Region B: sum greater than 4
 * - "C<6" → Region C: sum less than 6
 * - "D=" → Region D: all equal
 * - "Ex" or "E✕" → Region E: all different
 * - "F" → Region F: no constraint (removes existing)
 *
 * Multiple constraints can be space-separated: "A=8 B>4 C= Dx"
 */
export function parseConstraintShorthand(
  input: string
): Record<number, ConstraintDef> {
  const result: Record<number, ConstraintDef> = {};

  // Split by whitespace
  const parts = input.trim().split(/\s+/);

  for (const part of parts) {
    const parsed = parseSingleConstraint(part);
    if (parsed) {
      result[parsed.regionIndex] = parsed.constraint;
    }
  }

  return result;
}

interface ParsedConstraint {
  regionIndex: number;
  constraint: ConstraintDef;
}

function parseSingleConstraint(part: string): ParsedConstraint | null {
  if (!part || part.length < 1) return null;

  // First character should be a letter A-J
  const letter = part[0].toUpperCase();
  if (letter < 'A' || letter > 'J') return null;

  const regionIndex = letter.charCodeAt(0) - 'A'.charCodeAt(0);
  const rest = part.substring(1);

  // No rest = remove constraint
  if (!rest) {
    return { regionIndex, constraint: { type: 'none' } };
  }

  // Check for all_equal: just "=" with no number
  if (rest === '=') {
    return { regionIndex, constraint: { type: 'all_equal' } };
  }

  // Check for all_different: "x" or "✕"
  if (rest.toLowerCase() === 'x' || rest === '✕') {
    return { regionIndex, constraint: { type: 'all_different' } };
  }

  // Parse sum constraints: "=8", ">4", "<6", "!=3", or just "8"
  const sumMatch = rest.match(/^(==?|!=|[<>])?(\d+)$/);
  if (sumMatch) {
    const [, opStr, valueStr] = sumMatch;
    let op: ConstraintOp = '==';

    if (opStr === '>' ) op = '>';
    else if (opStr === '<') op = '<';
    else if (opStr === '!=' || opStr === '≠') op = '!=';
    // "=" or "==" or no op all mean equals

    return {
      regionIndex,
      constraint: {
        type: 'sum',
        op,
        value: parseInt(valueStr, 10),
      },
    };
  }

  return null;
}

/**
 * Convert constraint definitions to shorthand string
 */
export function constraintsToShorthand(
  constraints: Record<number, ConstraintDef>
): string {
  const parts: string[] = [];

  for (const [indexStr, constraint] of Object.entries(constraints)) {
    const regionIndex = parseInt(indexStr, 10);
    const letter = String.fromCharCode('A'.charCodeAt(0) + regionIndex);

    if (constraint.type === 'none') {
      continue;
    } else if (constraint.type === 'all_equal') {
      parts.push(`${letter}=`);
    } else if (constraint.type === 'all_different') {
      parts.push(`${letter}x`);
    } else if (constraint.type === 'sum') {
      const op = constraint.op === '==' ? '=' : constraint.op;
      parts.push(`${letter}${op}${constraint.value}`);
    }
  }

  return parts.join(' ');
}

/**
 * Get a display label for a constraint
 */
export function getConstraintLabel(constraint: ConstraintDef | undefined): string {
  if (!constraint || constraint.type === 'none') {
    return '—';
  }

  if (constraint.type === 'all_equal') {
    return '=';
  }

  if (constraint.type === 'all_different') {
    return '✕';
  }

  if (constraint.type === 'sum') {
    const opSymbol = {
      '==': '',
      '!=': '≠',
      '<': '<',
      '>': '>',
    }[constraint.op || '=='];
    return `${opSymbol}${constraint.value}`;
  }

  return '—';
}

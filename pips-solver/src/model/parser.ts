/**
 * Parser for YAML/JSON puzzle specifications
 */

import YAML from 'yaml';
import { PuzzleSpec } from './types';

export interface ParseResult {
  success: boolean;
  spec?: PuzzleSpec;
  error?: string;
}

/**
 * Parse a YAML or JSON string into a PuzzleSpec
 */
export function parsePuzzle(input: string): ParseResult {
  try {
    // Try parsing as JSON first
    let data: any;
    try {
      data = JSON.parse(input);
    } catch {
      // If JSON fails, try YAML
      data = YAML.parse(input);
    }

    if (!data) {
      return {
        success: false,
        error: 'Empty puzzle specification',
      };
    }

    // Validate required fields
    if (typeof data.rows !== 'number' || data.rows <= 0) {
      return {
        success: false,
        error: 'Missing or invalid "rows" field',
      };
    }

    if (typeof data.cols !== 'number' || data.cols <= 0) {
      return {
        success: false,
        error: 'Missing or invalid "cols" field',
      };
    }

    if (!Array.isArray(data.regions)) {
      return {
        success: false,
        error: 'Missing or invalid "regions" field (must be 2D array)',
      };
    }

    // Validate regions array dimensions
    if (data.regions.length !== data.rows) {
      return {
        success: false,
        error: `regions array must have ${data.rows} rows, got ${data.regions.length}`,
      };
    }

    for (let r = 0; r < data.rows; r++) {
      if (!Array.isArray(data.regions[r])) {
        return {
          success: false,
          error: `regions[${r}] must be an array`,
        };
      }
      if (data.regions[r].length !== data.cols) {
        return {
          success: false,
          error: `regions[${r}] must have ${data.cols} columns, got ${data.regions[r].length}`,
        };
      }
    }

    // Parse constraints
    const constraints: { [regionId: number]: any } = {};
    if (data.constraints) {
      if (typeof data.constraints !== 'object') {
        return {
          success: false,
          error: 'constraints must be an object',
        };
      }

      for (const [key, value] of Object.entries(data.constraints)) {
        const regionId = parseInt(key, 10);
        if (isNaN(regionId)) {
          return {
            success: false,
            error: `Invalid region ID in constraints: ${key}`,
          };
        }

        const constraint = value as any;

        // Validate constraint structure
        if (constraint.sum !== undefined && typeof constraint.sum !== 'number') {
          return {
            success: false,
            error: `Invalid sum constraint for region ${regionId}`,
          };
        }

        if (constraint.op !== undefined) {
          if (!['=', '<', '>', '≠'].includes(constraint.op)) {
            return {
              success: false,
              error: `Invalid operator "${constraint.op}" for region ${regionId}`,
            };
          }
          if (constraint.value === undefined || typeof constraint.value !== 'number') {
            return {
              success: false,
              error: `Operator constraint requires a "value" for region ${regionId}`,
            };
          }
        }

        if (constraint.all_equal !== undefined && typeof constraint.all_equal !== 'boolean') {
          return {
            success: false,
            error: `all_equal must be boolean for region ${regionId}`,
          };
        }

        constraints[regionId] = constraint;
      }
    }

    const spec: PuzzleSpec = {
      id: data.id || generateId(),
      name: data.name || 'Untitled Puzzle',
      rows: data.rows,
      cols: data.cols,
      maxPip: data.maxPip !== undefined ? data.maxPip : 6,
      allowDuplicates: data.allowDuplicates !== undefined ? data.allowDuplicates : false,
      regions: data.regions,
      constraints,
    };

    return {
      success: true,
      spec,
    };
  } catch (error) {
    return {
      success: false,
      error: `Parse error: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

/**
 * Convert a PuzzleSpec to YAML string
 */
export function specToYAML(spec: PuzzleSpec): string {
  return YAML.stringify({
    id: spec.id,
    name: spec.name,
    rows: spec.rows,
    cols: spec.cols,
    maxPip: spec.maxPip,
    allowDuplicates: spec.allowDuplicates,
    regions: spec.regions,
    constraints: spec.constraints,
  });
}

/**
 * Convert a PuzzleSpec to JSON string
 */
export function specToJSON(spec: PuzzleSpec): string {
  return JSON.stringify(
    {
      id: spec.id,
      name: spec.name,
      rows: spec.rows,
      cols: spec.cols,
      maxPip: spec.maxPip,
      allowDuplicates: spec.allowDuplicates,
      regions: spec.regions,
      constraints: spec.constraints,
    },
    null,
    2
  );
}

function generateId(): string {
  return `puzzle_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

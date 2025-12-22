/**
 * JSON Parsing Utilities for AI Extraction
 * Handles common issues with LLM-generated JSON (multiline strings, missing quotes, etc.)
 */

import { z } from 'zod';

// ════════════════════════════════════════════════════════════════════════════
// JSON Extraction
// ════════════════════════════════════════════════════════════════════════════

/**
 * Extracts JSON object from text that may contain markdown or other content.
 * Handles common LLM output issues like multiline strings in shape/regions fields.
 *
 * @param text - Raw text response from LLM
 * @returns Cleaned JSON string or null if no JSON found
 */
export function extractJSON(text: string): string | null {
  // Try to find JSON object in response
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return null;

  let jsonStr = match[0];

  // Fix multiline string issues in shape/regions fields
  // LLMs sometimes output these as multiple quoted strings on separate lines
  jsonStr = fixMultilineField('shape', jsonStr);
  jsonStr = fixMultilineField('regions', jsonStr);

  return jsonStr;
}

/**
 * Fixes multiline string fields that LLMs sometimes output incorrectly.
 * Converts patterns like:
 *   "shape": "....."
 *             "....."
 * To:
 *   "shape": ".....\n....."
 *
 * @param fieldName - The JSON field name to fix
 * @param json - The JSON string to process
 * @returns Fixed JSON string
 */
export function fixMultilineField(fieldName: string, json: string): string {
  // Pattern: "fieldName": "content"
  //          "more content" (indented)
  const pattern = new RegExp(`"${fieldName}"\\s*:\\s*"([^"]*)"\\s*\\n\\s*"([^"]*)"`, 'g');

  let fixed = json;
  let prevFixed = '';
  let iterations = 0;

  // Keep fixing until no more matches (handles multiple lines)
  while (fixed !== prevFixed && iterations < 10) {
    prevFixed = fixed;
    fixed = fixed.replace(pattern, (_, line1, line2) => {
      // Join with newline character
      return `"${fieldName}": "${line1}\\n${line2}"`;
    });
    iterations++;
  }

  return fixed;
}

// ════════════════════════════════════════════════════════════════════════════
// Safe JSON Parsing with Zod Validation
// ════════════════════════════════════════════════════════════════════════════

export type ParseResult<T> =
  | { success: true; data: T }
  | { success: false; error: string };

/**
 * Safely parses JSON text and validates against a Zod schema.
 * Handles extraction of JSON from mixed content and common LLM output issues.
 *
 * @param text - Raw text that may contain JSON
 * @param schema - Zod schema to validate against
 * @returns Parse result with validated data or error message
 */
export function parseJSONSafely<T>(
  text: string,
  schema: z.ZodSchema<T>
): ParseResult<T> {
  const jsonStr = extractJSON(text);
  if (!jsonStr) {
    return { success: false, error: 'No JSON found in response' };
  }

  try {
    const parsed = JSON.parse(jsonStr);
    const validated = schema.safeParse(parsed);
    if (!validated.success) {
      const errors = validated.error.issues.map(
        (e: z.ZodIssue) => `${e.path.join('.')}: ${e.message}`
      );
      return { success: false, error: `Validation failed: ${errors.join(', ')}` };
    }
    return { success: true, data: validated.data };
  } catch (e) {
    return { success: false, error: `JSON parse error: ${e}` };
  }
}

/**
 * Attempts to parse JSON with fallback strategies for complex multiline fields.
 * More aggressive fixing for shape/regions that span 3-4+ lines.
 *
 * @param jsonMatch - The matched JSON string
 * @param schema - Zod schema to validate against
 * @returns Parse result with validated data or error message
 */
export function parseJSONWithFallback<T>(
  jsonMatch: string,
  schema: z.ZodSchema<T>
): ParseResult<T> {
  // First try standard extraction
  let jsonString = jsonMatch;
  jsonString = fixMultilineField('shape', jsonString);
  jsonString = fixMultilineField('regions', jsonString);

  try {
    const parsed = JSON.parse(jsonString);
    const validated = schema.safeParse(parsed);
    if (!validated.success) {
      const errors = validated.error.issues.map(
        (e: z.ZodIssue) => `${e.path.join('.')}: ${e.message}`
      );
      return { success: false, error: `Validation failed: ${errors.join(', ')}` };
    }
    return { success: true, data: validated.data };
  } catch (parseError) {
    // Fallback: try more aggressive fixing for multiline strings
    try {
      let fallbackFix = jsonMatch;

      // Handle 4 lines
      fallbackFix = fallbackFix.replace(
        /("(?:shape|regions)"\s*:\s*")([^"]*)"\s*\n\s*"([^"]*)"\s*\n\s*"([^"]*)"\s*\n\s*"([^"]*)"/g,
        (_, prefix, l1, l2, l3, l4) => {
          return `${prefix}${l1}\\n${l2}\\n${l3}\\n${l4}"`;
        }
      );
      // Handle 3 lines
      fallbackFix = fallbackFix.replace(
        /("(?:shape|regions)"\s*:\s*")([^"]*)"\s*\n\s*"([^"]*)"\s*\n\s*"([^"]*)"/g,
        (_, prefix, l1, l2, l3) => {
          return `${prefix}${l1}\\n${l2}\\n${l3}"`;
        }
      );
      // Handle 2 lines
      fallbackFix = fallbackFix.replace(
        /("(?:shape|regions)"\s*:\s*")([^"]*)"\s*\n\s*"([^"]*)"/g,
        (_, prefix, l1, l2) => {
          return `${prefix}${l1}\\n${l2}"`;
        }
      );

      const parsed = JSON.parse(fallbackFix);
      const validated = schema.safeParse(parsed);
      if (!validated.success) {
        const errors = validated.error.issues.map(
          (e: z.ZodIssue) => `${e.path.join('.')}: ${e.message}`
        );
        return { success: false, error: `Validation failed: ${errors.join(', ')}` };
      }
      return { success: true, data: validated.data };
    } catch (secondError) {
      return {
        success: false,
        error: `Invalid JSON: ${parseError}. Fallback attempt: ${secondError}`,
      };
    }
  }
}

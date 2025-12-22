/**
 * Stage 1: Grid Geometry Extraction
 *
 * Extracts grid dimensions (rows, cols) from NYT Pips screenshot.
 * Uses 3-model ensemble with consensus voting.
 */

import { ExtractionConfig, GridGeometryResult, ModelResponse } from '../types';
import { callAllModels } from '../apiClient';
import { NYT_VALIDATION } from '../config';

// =============================================================================
// Prompt Template
// =============================================================================

const GRID_GEOMETRY_PROMPT = `Analyze this NYT Pips puzzle screenshot and count the EXACT grid dimensions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP-BY-STEP COUNTING METHOD:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LOCATE THE PUZZLE GRID:
   - Find the main rectangular area with colored cells (NOT the domino tray at bottom)
   - The puzzle grid has cells in various colors (green, pink, yellow, blue, etc.)
   - Each cell may contain a small diamond with a number or symbol

2. COUNT ROWS (horizontal lines, top to bottom):
   - Start at the TOP of the grid
   - Count each horizontal level of cells going DOWN
   - Include rows even if they have holes/gaps (dark empty squares)

   Example of a 5-row grid:
   ┌─┬─┬─┬─┐  ← Row 1
   ├─┼─┼─┼─┤  ← Row 2
   ├─┼─┼─┼─┤  ← Row 3
   ├─┼─┼─┼─┤  ← Row 4
   └─┴─┴─┴─┘  ← Row 5

3. COUNT COLUMNS (vertical lines, left to right):
   - Start at the LEFT of the grid
   - Count each vertical column going RIGHT
   - Include columns even if they have holes/gaps

   Example of a 4-column grid:
   ↓ ↓ ↓ ↓
   1 2 3 4 (columns)
   ┌─┬─┬─┬─┐
   │ │ │ │ │
   └─┴─┴─┴─┘

4. HANDLE HOLES/IRREGULAR SHAPES:
   - Holes are dark/black empty squares WITHIN the grid boundary
   - Holes still count as grid positions!

   Example - This is still a 4×5 grid (4 rows, 5 cols):
   ┌─┬─┬─┬─┬─┐
   │▓│ │ │ │▓│  ← Row 1 (has 2 holes at corners)
   │ │ │ │ │ │  ← Row 2
   │ │ │ │ │ │  ← Row 3
   │▓│ │ │ │▓│  ← Row 4 (has 2 holes at corners)
   └─┴─┴─┴─┴─┘
   (▓ = hole, counts as grid position)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON MISTAKES TO AVOID:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✗ DON'T count only visible/colored cells (holes are positions too!)
✗ DON'T include the domino tray (row of dominoes shown below the puzzle)
✗ DON'T confuse grid lines with cell content
✗ DON'T count constraint symbols or numbers as additional cells

✓ DO count from the outermost boundaries of the puzzle grid
✓ DO include corner holes in your dimension count
✓ DO verify by multiplying: rows × cols should roughly match visible cells + holes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED RANGES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- NYT Pips puzzles typically have 4-8 rows and 4-8 columns
- Common sizes: 5×5, 6×5, 5×6, 6×6, 7×5, 5×7

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY this JSON (no markdown, no explanation, no code blocks):
{"rows": N, "cols": M, "confidence": 0.XX}

Confidence scoring:
- 0.95-1.00: Grid boundaries crystal clear, 100% certain of count
- 0.85-0.94: Very confident, only minor visual ambiguity
- 0.70-0.84: Moderately confident, some edges unclear
- Below 0.70: Low confidence, significant uncertainty`;

// =============================================================================
// Retry Prompt (with clarification)
// =============================================================================

function getRetryPrompt(previousAttempts: GridGeometryResult[]): string {
  const attemptsStr = previousAttempts
    .map((a, i) => `Model ${i + 1}: ${a.rows}×${a.cols}`)
    .join(', ');

  // Calculate if there's a close disagreement (off by 1)
  const rows = previousAttempts.map(a => a.rows);
  const cols = previousAttempts.map(a => a.cols);
  const rowRange = rows.length > 0 ? Math.max(...rows) - Math.min(...rows) : 0;
  const colRange = cols.length > 0 ? Math.max(...cols) - Math.min(...cols) : 0;

  const offByOneHint = (rowRange === 1 || colRange === 1)
    ? `\n⚠️ Models differ by just 1 - VERY carefully verify the boundary edges!`
    : '';

  return `⚠️ RE-EXAMINE: Previous extractions disagreed on grid dimensions.

PREVIOUS RESULTS: ${attemptsStr}${offByOneHint}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRECISE COUNTING TECHNIQUE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. IDENTIFY GRID BOUNDARIES:
   - Find the TOP-MOST row of cells (may have holes at corners)
   - Find the BOTTOM-MOST row of cells
   - Find the LEFT-MOST column of cells
   - Find the RIGHT-MOST column of cells

2. COUNT SYSTEMATICALLY:
   ROWS: Point to each horizontal row from top to bottom, count aloud:
         "Row 1... Row 2... Row 3..." etc.

   COLS: Point to each vertical column from left to right, count aloud:
         "Col 1... Col 2... Col 3..." etc.

3. VERIFY YOUR COUNT:
   - A ${Math.min(...rows)}×${Math.min(...cols)} grid has ${Math.min(...rows) * Math.min(...cols)} total positions
   - A ${Math.max(...rows)}×${Math.max(...cols)} grid has ${Math.max(...rows) * Math.max(...cols)} total positions
   - Which matches the actual grid better when counting cells + holes?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL REMINDERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Holes (black/dark squares) ARE grid positions - count them!
• The domino tray below the puzzle is NOT part of the grid
• Corner holes still define the grid boundary

Example - Both outer edges have holes, but it's still 6×5:
  ┌─┬─┬─┬─┬─┐
  │▓│ │ │ │▓│  ← Still Row 1
  │ │ │ │ │ │
  │ │ │ │ │ │
  │ │ │ │ │ │
  │ │ │ │ │ │
  │▓│ │ │ │▓│  ← Still Row 6
  └─┴─┴─┴─┴─┘
   ↑         ↑
  Col 1    Col 5

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY this JSON (no markdown, no explanation):
{"rows": N, "cols": M, "confidence": 0.XX}`;
}

// =============================================================================
// Response Parser
// =============================================================================

function parseGridResponse(content: string): GridGeometryResult | null {
  try {
    // Extract JSON from response (handle markdown code blocks)
    let jsonStr = content.trim();

    // Remove markdown code block if present
    if (jsonStr.startsWith('```')) {
      jsonStr = jsonStr.replace(/```json?\n?/g, '').replace(/```/g, '').trim();
    }

    // Find JSON object
    const jsonMatch = jsonStr.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return null;
    }

    const parsed = JSON.parse(jsonMatch[0]);

    // Validate required fields
    if (
      typeof parsed.rows !== 'number' ||
      typeof parsed.cols !== 'number' ||
      typeof parsed.confidence !== 'number'
    ) {
      return null;
    }

    return {
      rows: Math.round(parsed.rows),
      cols: Math.round(parsed.cols),
      confidence: Math.min(1, Math.max(0, parsed.confidence)),
    };
  } catch {
    return null;
  }
}

// =============================================================================
// Validation
// =============================================================================

function validateGridResult(result: GridGeometryResult): string[] {
  const errors: string[] = [];
  const { minGridSize, maxGridSize } = NYT_VALIDATION;

  if (result.rows < minGridSize || result.rows > maxGridSize) {
    errors.push(`Rows ${result.rows} outside valid range [${minGridSize}, ${maxGridSize}]`);
  }

  if (result.cols < minGridSize || result.cols > maxGridSize) {
    errors.push(`Cols ${result.cols} outside valid range [${minGridSize}, ${maxGridSize}]`);
  }

  return errors;
}

// =============================================================================
// Main Extraction Function
// =============================================================================

export interface GridGeometryStageResult {
  result: GridGeometryResult;
  responses: ModelResponse<GridGeometryResult>[];
  retryCount: number;
  validationErrors: string[];
}

/**
 * Extract grid geometry from image using 3-model ensemble
 */
export async function extractGridGeometry(
  imageBase64: string,
  config: ExtractionConfig
): Promise<GridGeometryStageResult> {
  const allResponses: ModelResponse<GridGeometryResult>[] = [];
  let retryCount = 0;

  // Initial extraction
  let responses = await callModelsForGrid(imageBase64, GRID_GEOMETRY_PROMPT, config);
  allResponses.push(...responses);

  // Check if we need to retry
  const validResponses = responses.filter((r) => r.answer !== null && !r.error);

  if (validResponses.length < 2 && retryCount < config.maxRetries) {
    // Not enough valid responses, retry
    retryCount++;
    const retryResponses = await callModelsForGrid(
      imageBase64,
      getRetryPrompt(validResponses.map((r) => r.answer!)),
      config
    );
    allResponses.push(...retryResponses);
    responses = [...validResponses, ...retryResponses.filter((r) => r.answer !== null && !r.error)];
  }

  // Select best result using consensus
  const result = selectBestResult(responses.filter((r) => r.answer !== null) as ModelResponse<GridGeometryResult>[]);
  const validationErrors = validateGridResult(result);

  return {
    result,
    responses: allResponses,
    retryCount,
    validationErrors,
  };
}

// =============================================================================
// Helper Functions
// =============================================================================

async function callModelsForGrid(
  imageBase64: string,
  prompt: string,
  config: ExtractionConfig
): Promise<ModelResponse<GridGeometryResult>[]> {
  const apiResponses = await callAllModels(imageBase64, prompt, config);
  const results: ModelResponse<GridGeometryResult>[] = [];

  console.log(`[GridGeometry] Received ${apiResponses.size} API responses`);

  for (const [model, response] of apiResponses) {
    console.log(`[GridGeometry] Model ${model}:`, {
      hasError: !!response.error,
      error: response.error,
      contentLength: response.content?.length ?? 0,
      contentPreview: response.content?.substring(0, 200),
    });

    const parsed = response.error ? null : parseGridResponse(response.content);

    if (!parsed && !response.error) {
      console.log(`[GridGeometry] Parse failed for ${model}. Raw content:`, response.content?.substring(0, 500));
    }

    results.push({
      model,
      answer: parsed as GridGeometryResult,
      confidence: parsed?.confidence ?? 0,
      latencyMs: response.latencyMs,
      rawResponse: response.content,
      error: response.error || (parsed === null ? 'Failed to parse response' : undefined),
    });
  }

  const validCount = results.filter(r => r.answer !== null).length;
  console.log(`[GridGeometry] Valid responses: ${validCount}/${results.length}`);

  return results;
}

function selectBestResult(responses: ModelResponse<GridGeometryResult>[]): GridGeometryResult {
  if (responses.length === 0) {
    // Fallback if no valid responses
    console.warn('[GridGeometry] FALLBACK USED: No valid responses from any model. Returning default 6x5 grid.');
    return { rows: 6, cols: 5, confidence: 0 };
  }

  if (responses.length === 1) {
    return responses[0].answer;
  }

  // Sort by confidence
  const sorted = [...responses].sort((a, b) => b.confidence - a.confidence);
  const top = sorted[0];
  const second = sorted[1];

  // If top confidence is significantly higher (>0.10), use it
  if (top.confidence - second.confidence > 0.10) {
    return top.answer;
  }

  // Check for majority vote on dimensions
  const votes = new Map<string, { count: number; result: GridGeometryResult; totalConfidence: number }>();

  for (const r of responses) {
    const key = `${r.answer.rows}x${r.answer.cols}`;
    const existing = votes.get(key);
    if (existing) {
      existing.count++;
      existing.totalConfidence += r.confidence;
    } else {
      votes.set(key, { count: 1, result: r.answer, totalConfidence: r.confidence });
    }
  }

  // Find majority (2+ votes)
  for (const vote of votes.values()) {
    if (vote.count >= 2) {
      return {
        ...vote.result,
        confidence: vote.totalConfidence / vote.count,
      };
    }
  }

  // No majority, use highest confidence
  return top.answer;
}

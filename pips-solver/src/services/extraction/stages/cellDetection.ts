/**
 * Stage 2: Cell/Hole Detection
 *
 * Identifies which grid positions are cells vs holes.
 * Uses 3-model ensemble with consensus voting.
 */

import { ExtractionConfig, CellDetectionResult, ModelResponse, GridGeometryResult } from '../types';
import { callAllModels } from '../apiClient';

// =============================================================================
// Prompt Template
// =============================================================================

function getCellDetectionPrompt(grid: GridGeometryResult): string {
  return `Analyze this NYT Pips puzzle screenshot and identify which positions are CELLS vs HOLES in the ${grid.rows}x${grid.cols} grid.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL IDENTIFICATION GUIDE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CELLS (use '.' in output):
┌────────────────────────────────────────┐
│ • Have COLORED backgrounds (green,     │
│   pink, yellow, blue, orange, etc.)    │
│ • May contain a white diamond with     │
│   a number or symbol inside            │
│ • Are part of the playable puzzle area │
│ • Look FILLED and BRIGHT               │
└────────────────────────────────────────┘

HOLES (use '#' in output):
┌────────────────────────────────────────┐
│ • Are DARK or BLACK empty spaces       │
│ • Have NO colored background           │
│ • Are WITHIN the grid boundary         │
│ • Look EMPTY like gaps or cutouts      │
│ • Often appear at corners or edges     │
│ • May have subtle grid lines but       │
│   no fill color                        │
└────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP-BY-STEP DETECTION METHOD:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SCAN ROW BY ROW (top to bottom):
   - For each of the ${grid.rows} rows, examine ${grid.cols} positions left to right
   - Ask: "Does this position have a colored background?"
   - YES (colored) → write '.'
   - NO (dark/empty) → write '#'

2. EXAMPLE - 5x6 grid with corner holes:

   Visual appearance:          Output shape:
   ┌─┬─┬─┬─┬─┬─┐
   │▓│▓│█│█│█│█│  Row 1       ##....
   ├─┼─┼─┼─┼─┼─┤
   │█│█│█│█│█│█│  Row 2       ......
   ├─┼─┼─┼─┼─┼─┤
   │█│█│█│█│█│█│  Row 3       ......
   ├─┼─┼─┼─┼─┼─┤
   │█│█│█│█│█│█│  Row 4       ......
   ├─┼─┼─┼─┼─┼─┤
   │█│█│█│█│▓│▓│  Row 5       ....##
   └─┴─┴─┴─┴─┴─┘

   Legend: █ = colored cell (.), ▓ = dark hole (#)

3. EXAMPLE - 6x5 grid with L-shape holes:

   Visual appearance:          Output shape:
   ┌─┬─┬─┬─┬─┐
   │█│█│█│█│▓│  Row 1         ....#
   ├─┼─┼─┼─┼─┤
   │█│█│█│█│▓│  Row 2         ....#
   ├─┼─┼─┼─┼─┤
   │█│█│█│█│█│  Row 3         .....
   ├─┼─┼─┼─┼─┤
   │█│█│█│█│█│  Row 4         .....
   ├─┼─┼─┼─┼─┤
   │▓│█│█│█│█│  Row 5         #....
   ├─┼─┼─┼─┼─┤
   │▓│█│█│█│█│  Row 6         #....
   └─┴─┴─┴─┴─┘

4. EXAMPLE - 5x5 grid with scattered internal holes:

   Visual appearance:          Output shape:
   ┌─┬─┬─┬─┬─┐
   │█│█│█│█│█│  Row 1         .....
   ├─┼─┼─┼─┼─┤
   │█│▓│█│▓│█│  Row 2         .#.#.
   ├─┼─┼─┼─┼─┤
   │█│█│█│█│█│  Row 3         .....
   ├─┼─┼─┼─┼─┤
   │█│▓│█│▓│█│  Row 4         .#.#.
   ├─┼─┼─┼─┼─┤
   │█│█│█│█│█│  Row 5         .....
   └─┴─┴─┴─┴─┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON MISTAKES TO AVOID:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✗ DON'T confuse SHADOWS with holes (shadows are on colored cells, holes have NO color)
✗ DON'T confuse DARK-COLORED cells with holes (dark green/blue ARE cells, not holes)
✗ DON'T mark cells as holes just because they have constraint diamonds
✗ DON'T include the DOMINO TRAY (row of dominoes below puzzle) in your analysis
✗ DON'T count areas OUTSIDE the grid boundaries

✓ DO look for the absence of ANY background color = hole
✓ DO include corner holes that define irregular shapes
✓ DO count internal holes (gaps within the puzzle, not just edges)
✓ DO verify: total cells (.) should be EVEN (dominoes need pairs)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Output EXACTLY ${grid.rows} lines
- Each line has EXACTLY ${grid.cols} characters
- ONLY use '.' and '#' characters
- Total cell count (.) MUST be EVEN (for domino pairs)
- Typical puzzles have 0-8 holes, rarely more

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY this JSON (no markdown, no explanation, no code blocks):
{"shape": "line1\\nline2\\nline3\\n...", "confidence": 0.XX}

Confidence scoring:
- 0.95-1.00: All cell/hole boundaries crystal clear
- 0.85-0.94: Very confident, only minor visual ambiguity
- 0.70-0.84: Some positions unclear (shadows, dark colors)
- Below 0.70: Multiple positions ambiguous`;
}

// =============================================================================
// Retry Prompt
// =============================================================================

function getRetryPrompt(
  grid: GridGeometryResult,
  previousAttempts: CellDetectionResult[]
): string {
  const attemptsStr = previousAttempts
    .map((a, i) => `Model ${i + 1}:\n${a.shape}`)
    .join('\n\n');

  // Calculate differences to provide hints
  const shapes = previousAttempts.map(a => a.shape);
  const holeCountHint = shapes.length > 0
    ? `Hole counts ranged from ${Math.min(...shapes.map(s => (s.match(/#/g) || []).length))} to ${Math.max(...shapes.map(s => (s.match(/#/g) || []).length))}`
    : '';

  return `⚠️ RE-EXAMINE: Previous cell/hole detection attempts DISAGREED for this ${grid.rows}x${grid.cols} grid.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREVIOUS CONFLICTING RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

${attemptsStr}

${holeCountHint}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRECISE IDENTIFICATION TECHNIQUE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CELL vs HOLE - THE KEY QUESTION:
   For each grid position, ask: "Does this square have ANY background color?"

   CELL (.) answers:
   • "Yes, it's green/pink/yellow/blue/orange/purple"
   • "Yes, even though it looks dark, there's color there"
   • "Yes, it has a constraint diamond on a colored background"

   HOLE (#) answers:
   • "No, it's completely black/dark with no color"
   • "No, it's an empty cutout in the grid"
   • "No, it looks like a gap, not a playable space"

2. SCAN SYSTEMATICALLY:
   Row 1: [position 1] [position 2] ... [position ${grid.cols}]
   Row 2: [position 1] [position 2] ... [position ${grid.cols}]
   ...continue for all ${grid.rows} rows...

3. VISUAL COMPARISON:

   This is a CELL (colored):     This is a HOLE (empty):
   ┌─────────────────────┐       ┌─────────────────────┐
   │   ████████████████  │       │                     │
   │   █  COLORED BG  █  │       │   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒  │
   │   █  maybe with  █  │       │   ▒ DARK/BLACK ▒  │
   │   █  ◇ diamond  █  │       │   ▒  NO COLOR   ▒  │
   │   ████████████████  │       │   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒  │
   └─────────────────────┘       └─────────────────────┘
        Output: .                     Output: #

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL REMINDERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Dark-COLORED cells (dark blue, dark green) are still CELLS, not holes
• Shadows on colored cells do NOT make them holes
• Constraint diamonds (◇ with numbers) sit ON cells, not holes
• Holes have ZERO color - they are pure gaps in the puzzle
• Verify: count of '.' characters must be EVEN

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Exactly ${grid.rows} lines in output
- Each line exactly ${grid.cols} characters
- Only '.' and '#' characters allowed
- Total '.' count MUST be even (dominoes need pairs)

Return ONLY valid JSON (no markdown, no explanation):
{"shape": "line1\\nline2\\n...", "confidence": 0.XX}`;
}

// =============================================================================
// Response Parser
// =============================================================================

function parseCellResponse(content: string, grid: GridGeometryResult): CellDetectionResult | null {
  try {
    // Extract JSON from response
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

    if (typeof parsed.shape !== 'string' || typeof parsed.confidence !== 'number') {
      return null;
    }

    // Normalize shape string
    let shape = parsed.shape.trim();

    // Handle escaped newlines
    shape = shape.replace(/\\n/g, '\n');

    // Validate dimensions
    const lines = shape.split('\n').filter((l: string) => l.length > 0);

    if (lines.length !== grid.rows) {
      return null;
    }

    for (const line of lines) {
      if (line.length !== grid.cols) {
        return null;
      }
      if (!/^[.#]+$/.test(line)) {
        return null;
      }
    }

    return {
      shape: lines.join('\n'),
      confidence: Math.min(1, Math.max(0, parsed.confidence)),
    };
  } catch {
    return null;
  }
}

// =============================================================================
// Validation
// =============================================================================

function validateCellResult(result: CellDetectionResult, grid: GridGeometryResult): string[] {
  const errors: string[] = [];
  const lines = result.shape.split('\n');

  if (lines.length !== grid.rows) {
    errors.push(`Shape has ${lines.length} rows, expected ${grid.rows}`);
  }

  for (let i = 0; i < lines.length; i++) {
    if (lines[i].length !== grid.cols) {
      errors.push(`Row ${i + 1} has ${lines[i].length} cols, expected ${grid.cols}`);
    }
  }

  // Count cells - must be even for dominoes
  const cellCount = (result.shape.match(/\./g) || []).length;
  if (cellCount % 2 !== 0) {
    errors.push(`Cell count ${cellCount} is odd - dominoes need even count`);
  }

  if (cellCount < 8) {
    errors.push(`Only ${cellCount} cells - minimum viable puzzle needs 8+`);
  }

  return errors;
}

// =============================================================================
// Main Extraction Function
// =============================================================================

export interface CellDetectionStageResult {
  result: CellDetectionResult;
  responses: ModelResponse<CellDetectionResult>[];
  retryCount: number;
  validationErrors: string[];
}

/**
 * Extract cell/hole map from image using 3-model ensemble
 */
export async function extractCellDetection(
  imageBase64: string,
  grid: GridGeometryResult,
  config: ExtractionConfig
): Promise<CellDetectionStageResult> {
  const allResponses: ModelResponse<CellDetectionResult>[] = [];
  let retryCount = 0;

  // Initial extraction
  const prompt = getCellDetectionPrompt(grid);
  let responses = await callModelsForCells(imageBase64, prompt, grid, config);
  allResponses.push(...responses);

  // Check if we need to retry
  let validResponses = responses.filter((r) => r.answer !== null && !r.error);

  if (validResponses.length < 2 && retryCount < config.maxRetries) {
    retryCount++;
    const retryResponses = await callModelsForCells(
      imageBase64,
      getRetryPrompt(grid, validResponses.map((r) => r.answer!)),
      grid,
      config
    );
    allResponses.push(...retryResponses);
    validResponses = [
      ...validResponses,
      ...retryResponses.filter((r) => r.answer !== null && !r.error),
    ];
  }

  // Select best result using consensus
  const result = selectBestResult(
    validResponses as ModelResponse<CellDetectionResult>[],
    grid
  );
  const validationErrors = validateCellResult(result, grid);

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

async function callModelsForCells(
  imageBase64: string,
  prompt: string,
  grid: GridGeometryResult,
  config: ExtractionConfig
): Promise<ModelResponse<CellDetectionResult>[]> {
  const apiResponses = await callAllModels(imageBase64, prompt, config);
  const results: ModelResponse<CellDetectionResult>[] = [];

  console.log(`[CellDetection] Received ${apiResponses.size} API responses for ${grid.rows}x${grid.cols} grid`);

  for (const [model, response] of apiResponses) {
    console.log(`[CellDetection] Model ${model}:`, {
      hasError: !!response.error,
      error: response.error,
      contentLength: response.content?.length ?? 0,
    });

    const parsed = response.error ? null : parseCellResponse(response.content, grid);

    if (!parsed && !response.error) {
      console.log(`[CellDetection] Parse failed for ${model}. Grid: ${grid.rows}x${grid.cols}. Raw:`, response.content?.substring(0, 300));
    }

    results.push({
      model,
      answer: parsed as CellDetectionResult,
      confidence: parsed?.confidence ?? 0,
      latencyMs: response.latencyMs,
      rawResponse: response.content,
      error: response.error || (parsed === null ? 'Failed to parse response' : undefined),
    });
  }

  const validCount = results.filter(r => r.answer !== null).length;
  console.log(`[CellDetection] Valid responses: ${validCount}/${results.length}`);

  return results;
}

function selectBestResult(
  responses: ModelResponse<CellDetectionResult>[],
  grid: GridGeometryResult
): CellDetectionResult {
  if (responses.length === 0) {
    // Fallback: all cells, no holes
    console.warn(`[CellDetection] FALLBACK USED: No valid responses. Returning all-cells ${grid.rows}x${grid.cols} grid.`);
    const row = '.'.repeat(grid.cols);
    return {
      shape: Array(grid.rows).fill(row).join('\n'),
      confidence: 0,
    };
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

  // Check for majority vote on shape
  const votes = new Map<string, { count: number; result: CellDetectionResult; totalConfidence: number }>();

  for (const r of responses) {
    // Normalize shape for comparison
    const key = r.answer.shape.replace(/\s+/g, '');
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

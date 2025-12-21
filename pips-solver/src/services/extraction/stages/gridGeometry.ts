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

const GRID_GEOMETRY_PROMPT = `Count the puzzle grid dimensions in this NYT Pips screenshot.

INSTRUCTIONS:
1. Look at the main puzzle grid (colored cells arranged in rows and columns)
2. Count ROWS: horizontal lines of cells from top to bottom
3. Count COLS: vertical columns from left to right
4. Include holes (empty/dark positions within the grid bounds) in your count
5. Do NOT count the domino tray area - only the main puzzle grid

NYT Pips grids are typically 4-8 rows and 4-8 columns.

IMPORTANT:
- Count grid LINES, not just visible colored cells
- A 6x5 grid has 6 rows and 5 columns
- Holes count as grid positions

Return ONLY valid JSON (no markdown, no explanation):
{"rows": N, "cols": M, "confidence": 0.XX}

Confidence scoring:
- 0.95-1.00: Grid lines perfectly clear, certain of count
- 0.85-0.94: Very confident, minor ambiguity
- 0.70-0.84: Moderately confident, some cells unclear
- Below 0.70: Low confidence, grid partially obscured`;

// =============================================================================
// Retry Prompt (with clarification)
// =============================================================================

function getRetryPrompt(previousAttempts: GridGeometryResult[]): string {
  const attemptsStr = previousAttempts
    .map((a, i) => `Attempt ${i + 1}: ${a.rows}x${a.cols}`)
    .join(', ');

  return `Count the puzzle grid dimensions in this NYT Pips screenshot.

PREVIOUS ATTEMPTS DISAGREED: ${attemptsStr}

Please look again MORE CAREFULLY:
1. Count grid LINES/POSITIONS, not just colored cells
2. Include any dark/empty holes within the grid bounds
3. The grid boundary is where colored cells stop
4. Do NOT count the domino reference tray

Tips:
- If you see 5 rows of cells but one row has a gap, it's still 5 rows
- Count the MAXIMUM width/height of the grid area
- NYT grids are typically 4-8 in each dimension

Return ONLY valid JSON:
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

  // Accept single high-confidence response without retry (faster extraction)
  const hasHighConfidence = validResponses.some((r) => r.confidence >= 0.80);
  const needsRetry = validResponses.length === 0 ||
    (validResponses.length < 2 && !hasHighConfidence);

  if (needsRetry && retryCount < config.maxRetries) {
    // Not enough valid responses, retry
    retryCount++;
    console.log(`[GridGeometry] Retrying: ${validResponses.length} valid, hasHighConf=${hasHighConfidence}`);
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

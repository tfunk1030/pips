/**
 * Stage 5: Domino Extraction
 *
 * Extracts domino tiles from the reference tray area.
 * Uses 3-model ensemble with consensus voting.
 */

import {
  ExtractionConfig,
  DominoExtractionResult,
  ModelResponse,
  CellDetectionResult,
} from '../types';
import { callAllModels } from '../apiClient';
import { NYT_VALIDATION } from '../config';

// =============================================================================
// Prompt Template
// =============================================================================

function getDominoPrompt(cells: CellDetectionResult): string {
  // Calculate expected domino count from cell count
  const cellCount = (cells.shape.match(/\./g) || []).length;
  const expectedDominoes = cellCount / 2;

  return `Count the dominoes in the tray/reference area of this NYT Pips screenshot.

EXPECTED DOMINO COUNT: ${expectedDominoes} (based on ${cellCount} grid cells)

INSTRUCTIONS:
1. Find the domino tray (usually at bottom or side of puzzle)
2. Each domino has TWO halves with 0-6 pips (dots) each
3. Count the pips on EACH half carefully
4. List all dominoes as [left_pips, right_pips] pairs

PIP COUNTING:
- 0 pips = blank/empty half
- 1-6 pips = count the dots carefully
- Standard domino pip patterns:
  - 1: center dot
  - 2: diagonal corners
  - 3: diagonal line
  - 4: four corners
  - 5: four corners + center
  - 6: two columns of 3

CRITICAL RULES:
- You MUST find exactly ${expectedDominoes} dominoes
- Each pip value must be 0-6
- NYT uses UNIQUE dominoes (no duplicates like [3,3] and [3,3])
- Order within domino doesn't matter ([2,5] = [5,2])

Return ONLY valid JSON (no markdown, no explanation):
{
  "dominoes": [[0, 1], [2, 3], [4, 5], ...],
  "confidence": 0.XX
}

Confidence scoring:
- 0.95-1.00: All dominoes clearly visible, pips easy to count
- 0.85-0.94: Most dominoes clear, 1-2 slightly blurry
- 0.70-0.84: Some pip counts uncertain
- Below 0.70: Multiple dominoes hard to read`;
}

// =============================================================================
// Retry Prompt
// =============================================================================

function getRetryPrompt(
  cells: CellDetectionResult,
  previousAttempts: DominoExtractionResult[]
): string {
  const cellCount = (cells.shape.match(/\./g) || []).length;
  const expectedDominoes = cellCount / 2;

  const attemptsStr = previousAttempts
    .map((a, i) => `Attempt ${i + 1}: ${JSON.stringify(a.dominoes)} (${a.dominoes.length} dominoes)`)
    .join('\n');

  return `Count dominoes in the tray area of this NYT Pips screenshot.

EXPECTED: ${expectedDominoes} dominoes (for ${cellCount} cells)

PREVIOUS ATTEMPTS DISAGREED:
${attemptsStr}

Please look again MORE CAREFULLY:
1. Focus on the DOMINO TRAY area (not the main grid)
2. Count pips on EACH HALF of each domino
3. Double-check counts: 5 and 6 are often confused

COMMON MISTAKES:
- Missing a domino in the corner
- Confusing 5 pips with 6 pips
- Counting the same domino twice
- Missing blank (0) halves

PIP PATTERNS:
- 0: empty
- 1: single center dot
- 2: two diagonal dots
- 3: three diagonal dots
- 4: four corner dots
- 5: four corners + center
- 6: six dots (2 columns of 3)

Return ONLY valid JSON:
{
  "dominoes": [[pip1, pip2], ...],
  "confidence": 0.XX
}`;
}

// =============================================================================
// Response Parser
// =============================================================================

function parseDominoResponse(content: string): DominoExtractionResult | null {
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

    if (!Array.isArray(parsed.dominoes) || typeof parsed.confidence !== 'number') {
      return null;
    }

    // Validate and normalize dominoes
    const dominoes: [number, number][] = [];

    for (const d of parsed.dominoes) {
      if (!Array.isArray(d) || d.length !== 2) {
        continue;
      }

      const pip1 = Math.round(Number(d[0]));
      const pip2 = Math.round(Number(d[1]));

      if (
        pip1 >= NYT_VALIDATION.pipRange[0] &&
        pip1 <= NYT_VALIDATION.pipRange[1] &&
        pip2 >= NYT_VALIDATION.pipRange[0] &&
        pip2 <= NYT_VALIDATION.pipRange[1]
      ) {
        // Normalize: smaller pip first
        dominoes.push(pip1 <= pip2 ? [pip1, pip2] : [pip2, pip1]);
      }
    }

    return {
      dominoes,
      confidence: Math.min(1, Math.max(0, parsed.confidence)),
    };
  } catch {
    return null;
  }
}

// =============================================================================
// Validation
// =============================================================================

function validateDominoResult(
  result: DominoExtractionResult,
  cells: CellDetectionResult
): string[] {
  const errors: string[] = [];
  const cellCount = (cells.shape.match(/\./g) || []).length;
  const expectedDominoes = cellCount / 2;

  // Check count
  if (result.dominoes.length !== expectedDominoes) {
    errors.push(
      `Found ${result.dominoes.length} dominoes, expected ${expectedDominoes} (for ${cellCount} cells)`
    );
  }

  // Check for duplicates
  const seen = new Set<string>();
  for (const [p1, p2] of result.dominoes) {
    // Normalize for comparison
    const key = `${Math.min(p1, p2)}-${Math.max(p1, p2)}`;
    if (seen.has(key)) {
      errors.push(`Duplicate domino: [${p1}, ${p2}]`);
    }
    seen.add(key);
  }

  // Check pip ranges
  for (const [p1, p2] of result.dominoes) {
    if (p1 < 0 || p1 > 6 || p2 < 0 || p2 > 6) {
      errors.push(`Invalid pip values: [${p1}, ${p2}] - must be 0-6`);
    }
  }

  return errors;
}

// =============================================================================
// Main Extraction Function
// =============================================================================

export interface DominoStageResult {
  result: DominoExtractionResult;
  responses: ModelResponse<DominoExtractionResult>[];
  retryCount: number;
  validationErrors: string[];
}

/**
 * Extract dominoes from image using 3-model ensemble
 */
export async function extractDominoes(
  imageBase64: string,
  cells: CellDetectionResult,
  config: ExtractionConfig
): Promise<DominoStageResult> {
  const allResponses: ModelResponse<DominoExtractionResult>[] = [];
  let retryCount = 0;

  // Initial extraction
  const prompt = getDominoPrompt(cells);
  let responses = await callModelsForDominoes(imageBase64, prompt, config);
  allResponses.push(...responses);

  // Check if we need to retry
  let validResponses = responses.filter((r) => r.answer !== null && !r.error);

  // Accept single high-confidence response without retry (faster extraction)
  const hasHighConfidence = validResponses.some((r) => r.confidence >= 0.80);
  const needsRetry = validResponses.length === 0 ||
    (validResponses.length < 2 && !hasHighConfidence);

  if (needsRetry && retryCount < config.maxRetries) {
    retryCount++;
    console.log(`[DominoExtraction] Retrying: ${validResponses.length} valid, hasHighConf=${hasHighConfidence}`);
    const retryResponses = await callModelsForDominoes(
      imageBase64,
      getRetryPrompt(cells, validResponses.map((r) => r.answer!)),
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
    validResponses as ModelResponse<DominoExtractionResult>[],
    cells
  );
  const validationErrors = validateDominoResult(result, cells);

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

async function callModelsForDominoes(
  imageBase64: string,
  prompt: string,
  config: ExtractionConfig
): Promise<ModelResponse<DominoExtractionResult>[]> {
  const apiResponses = await callAllModels(imageBase64, prompt, config);
  const results: ModelResponse<DominoExtractionResult>[] = [];

  for (const [model, response] of apiResponses) {
    const parsed = response.error ? null : parseDominoResponse(response.content);

    results.push({
      model,
      answer: parsed as DominoExtractionResult,
      confidence: parsed?.confidence ?? 0,
      latencyMs: response.latencyMs,
      rawResponse: response.content,
      error: response.error || (parsed === null ? 'Failed to parse response' : undefined),
    });
  }

  return results;
}

function selectBestResult(
  responses: ModelResponse<DominoExtractionResult>[],
  cells: CellDetectionResult
): DominoExtractionResult {
  const cellCount = (cells.shape.match(/\./g) || []).length;
  const expectedDominoes = cellCount / 2;

  if (responses.length === 0) {
    return { dominoes: [], confidence: 0 };
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

  // Prefer results with correct count
  const withCorrectCount = responses.filter(
    (r) => r.answer.dominoes.length === expectedDominoes
  );

  if (withCorrectCount.length > 0) {
    // Among correct count, use highest confidence
    withCorrectCount.sort((a, b) => b.confidence - a.confidence);
    return withCorrectCount[0].answer;
  }

  // Per-domino voting
  const allDominoes = new Map<string, { count: number; domino: [number, number] }>();

  for (const r of responses) {
    for (const [p1, p2] of r.answer.dominoes) {
      const key = `${Math.min(p1, p2)}-${Math.max(p1, p2)}`;
      const existing = allDominoes.get(key);
      if (existing) {
        existing.count++;
      } else {
        allDominoes.set(key, {
          count: 1,
          domino: [Math.min(p1, p2), Math.max(p1, p2)] as [number, number],
        });
      }
    }
  }

  // Take dominoes that appear in 2+ responses
  const consensusDominoes: [number, number][] = [];
  for (const vote of allDominoes.values()) {
    if (vote.count >= 2) {
      consensusDominoes.push(vote.domino);
    }
  }

  // If we have enough consensus dominoes, use them
  if (consensusDominoes.length >= expectedDominoes * 0.8) {
    return {
      dominoes: consensusDominoes.slice(0, expectedDominoes),
      confidence: (top.confidence + second.confidence) / 2,
    };
  }

  // Fall back to highest confidence
  return top.answer;
}

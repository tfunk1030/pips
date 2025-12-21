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
  return `Given this ${grid.rows}x${grid.cols} NYT Pips puzzle grid, identify which positions are cells vs holes.

DEFINITIONS:
- Cell (.): A colored background position where a domino pip can be placed
- Hole (#): An empty/dark space where no cell exists (often at corners or edges)

INSTRUCTIONS:
1. Examine each position in the ${grid.rows}x${grid.cols} grid
2. Build a shape string with ${grid.rows} lines, each with ${grid.cols} characters
3. Use '.' for cells (colored), '#' for holes (dark/empty)

EXAMPLE for a 5x6 grid with corner holes:
##....
......
......
......
....##

CRITICAL RULES:
- Output EXACTLY ${grid.rows} lines
- Each line has EXACTLY ${grid.cols} characters
- ONLY use '.' and '#' characters
- Total '.' count must be EVEN (dominoes need pairs)

Return ONLY valid JSON (no markdown, no explanation):
{"shape": "line1\\nline2\\n...", "confidence": 0.XX}

Confidence scoring:
- 0.95-1.00: All cell/hole boundaries perfectly clear
- 0.85-0.94: Very confident, minor edge ambiguity
- 0.70-0.84: Some positions unclear
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
    .map((a, i) => `Attempt ${i + 1}:\n${a.shape}`)
    .join('\n\n');

  return `Given this ${grid.rows}x${grid.cols} NYT Pips puzzle grid, identify cells vs holes.

PREVIOUS ATTEMPTS DISAGREED:
${attemptsStr}

Please look again MORE CAREFULLY:
1. Colored/bright areas = cells (.)
2. Dark/black/empty areas within grid bounds = holes (#)
3. The grid is EXACTLY ${grid.rows} rows by ${grid.cols} columns

VALIDATION RULES:
- Exactly ${grid.rows} lines in output
- Each line exactly ${grid.cols} characters
- Only '.' and '#' allowed
- Total cells must be even (for domino pairs)

Common mistakes to avoid:
- Counting areas OUTSIDE the grid
- Mistaking shadows for holes
- Missing small corner holes

Return ONLY valid JSON:
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

  for (const [model, response] of apiResponses) {
    const parsed = response.error ? null : parseCellResponse(response.content, grid);

    results.push({
      model,
      answer: parsed as CellDetectionResult,
      confidence: parsed?.confidence ?? 0,
      latencyMs: response.latencyMs,
      rawResponse: response.content,
      error: response.error || (parsed === null ? 'Failed to parse response' : undefined),
    });
  }

  return results;
}

function selectBestResult(
  responses: ModelResponse<CellDetectionResult>[],
  grid: GridGeometryResult
): CellDetectionResult {
  if (responses.length === 0) {
    // Fallback: all cells, no holes
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

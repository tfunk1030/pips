/**
 * Stage 3: Region Mapping
 *
 * Identifies distinct colored regions and assigns labels (A-Z).
 * Uses 3-model ensemble with consensus voting.
 */

import {
  ExtractionConfig,
  RegionMappingResult,
  ModelResponse,
  GridGeometryResult,
  CellDetectionResult,
} from '../types';
import { callAllModels } from '../apiClient';

// =============================================================================
// Prompt Template
// =============================================================================

function getRegionMappingPrompt(grid: GridGeometryResult, cells: CellDetectionResult): string {
  return `Given this NYT Pips puzzle grid, identify the distinct colored regions.

GRID SHAPE (${grid.rows}x${grid.cols}):
${cells.shape}
(where '.' = cell, '#' = hole)

INSTRUCTIONS:
1. Each distinct background COLOR is a separate region
2. Assign labels A, B, C, D... to each region (in reading order, top-left to bottom-right)
3. Build a regions string matching the shape dimensions exactly
4. Use '#' for holes (same positions as the shape above)
5. Use A-Z letters for cell positions based on their color

COMMON NYT COLORS:
- Pink, Coral, Orange, Peach/Tan
- Teal, Cyan, Light Blue
- Gray, Olive, Green, Purple

EXAMPLE:
Shape:      Regions:
##....      ##AABB
......      CCAABB
......      CCDDEE
......      FFDDEE
....##      FFGG##

CRITICAL RULES:
- Output EXACTLY ${grid.rows} lines with ${grid.cols} characters each
- '#' positions MUST match the shape above exactly
- Every '.' in shape becomes a letter (A-Z) in regions
- Adjacent cells with SAME color get SAME letter
- Adjacent cells with DIFFERENT colors get DIFFERENT letters
- Each region should have at least 2 cells

Return ONLY valid JSON (no markdown, no explanation):
{"regions": "line1\\nline2\\n...", "confidence": 0.XX}

Confidence scoring:
- 0.95-1.00: All region boundaries perfectly clear
- 0.85-0.94: Very confident, colors distinct
- 0.70-0.84: Some color boundaries ambiguous
- Below 0.70: Multiple regions unclear`;
}

// =============================================================================
// Retry Prompt
// =============================================================================

function getRetryPrompt(
  grid: GridGeometryResult,
  cells: CellDetectionResult,
  previousAttempts: RegionMappingResult[]
): string {
  const attemptsStr = previousAttempts
    .map((a, i) => `Attempt ${i + 1}:\n${a.regions}`)
    .join('\n\n');

  return `Given this NYT Pips puzzle grid, identify the colored regions.

GRID SHAPE (${grid.rows}x${grid.cols}):
${cells.shape}

PREVIOUS ATTEMPTS DISAGREED:
${attemptsStr}

Please look again MORE CAREFULLY:
1. Focus on BACKGROUND COLORS, not the grid lines
2. Similar but different colors (coral vs pink, teal vs cyan) ARE different regions
3. Each letter = one continuous region of the SAME color
4. Regions do NOT need to be rectangular

VALIDATION RULES:
- Exactly ${grid.rows} lines, ${grid.cols} characters each
- '#' must match shape positions exactly
- Only A-Z letters for cells (no numbers, no '.')
- Each region needs at least 2 cells

Return ONLY valid JSON:
{"regions": "line1\\nline2\\n...", "confidence": 0.XX}`;
}

// =============================================================================
// Response Parser
// =============================================================================

function parseRegionResponse(
  content: string,
  grid: GridGeometryResult,
  cells: CellDetectionResult
): RegionMappingResult | null {
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

    if (typeof parsed.regions !== 'string' || typeof parsed.confidence !== 'number') {
      return null;
    }

    // Normalize regions string
    let regions = parsed.regions.trim();
    regions = regions.replace(/\\n/g, '\n');

    const regionLines = regions.split('\n').filter((l: string) => l.length > 0);
    const shapeLines = cells.shape.split('\n');

    // Validate dimensions
    if (regionLines.length !== grid.rows) {
      return null;
    }

    for (let i = 0; i < regionLines.length; i++) {
      if (regionLines[i].length !== grid.cols) {
        return null;
      }

      // Validate character by character
      for (let j = 0; j < regionLines[i].length; j++) {
        const shapeChar = shapeLines[i][j];
        const regionChar = regionLines[i][j];

        if (shapeChar === '#') {
          // Holes must match
          if (regionChar !== '#') {
            return null;
          }
        } else {
          // Cells must be A-Z
          if (!/[A-Z]/.test(regionChar)) {
            return null;
          }
        }
      }
    }

    return {
      regions: regionLines.join('\n'),
      confidence: Math.min(1, Math.max(0, parsed.confidence)),
    };
  } catch {
    return null;
  }
}

// =============================================================================
// Validation
// =============================================================================

function validateRegionResult(
  result: RegionMappingResult,
  grid: GridGeometryResult,
  cells: CellDetectionResult
): string[] {
  const errors: string[] = [];
  const regionLines = result.regions.split('\n');
  const shapeLines = cells.shape.split('\n');

  // Check dimensions
  if (regionLines.length !== grid.rows) {
    errors.push(`Regions has ${regionLines.length} rows, expected ${grid.rows}`);
  }

  // Count cells per region
  const regionCounts = new Map<string, number>();

  for (let i = 0; i < regionLines.length; i++) {
    if (regionLines[i].length !== grid.cols) {
      errors.push(`Row ${i + 1} has ${regionLines[i].length} cols, expected ${grid.cols}`);
    }

    for (let j = 0; j < regionLines[i].length; j++) {
      const regionChar = regionLines[i][j];
      const shapeChar = shapeLines[i]?.[j];

      // Check hole alignment
      if (shapeChar === '#' && regionChar !== '#') {
        errors.push(`Position (${i},${j}) is hole in shape but not in regions`);
      }

      if (regionChar !== '#') {
        regionCounts.set(regionChar, (regionCounts.get(regionChar) || 0) + 1);
      }
    }
  }

  // Each region should have at least 2 cells
  for (const [region, count] of regionCounts) {
    if (count < 2) {
      errors.push(`Region ${region} has only ${count} cell(s) - needs at least 2`);
    }
  }

  return errors;
}

// =============================================================================
// Main Extraction Function
// =============================================================================

export interface RegionMappingStageResult {
  result: RegionMappingResult;
  responses: ModelResponse<RegionMappingResult>[];
  retryCount: number;
  validationErrors: string[];
}

/**
 * Extract region mapping from image using 3-model ensemble
 */
export async function extractRegionMapping(
  imageBase64: string,
  grid: GridGeometryResult,
  cells: CellDetectionResult,
  config: ExtractionConfig
): Promise<RegionMappingStageResult> {
  const allResponses: ModelResponse<RegionMappingResult>[] = [];
  let retryCount = 0;

  // Initial extraction
  const prompt = getRegionMappingPrompt(grid, cells);
  let responses = await callModelsForRegions(imageBase64, prompt, grid, cells, config);
  allResponses.push(...responses);

  // Check if we need to retry
  let validResponses = responses.filter((r) => r.answer !== null && !r.error);

  if (validResponses.length < 2 && retryCount < config.maxRetries) {
    retryCount++;
    const retryResponses = await callModelsForRegions(
      imageBase64,
      getRetryPrompt(grid, cells, validResponses.map((r) => r.answer!)),
      grid,
      cells,
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
    validResponses as ModelResponse<RegionMappingResult>[],
    grid,
    cells
  );
  const validationErrors = validateRegionResult(result, grid, cells);

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

async function callModelsForRegions(
  imageBase64: string,
  prompt: string,
  grid: GridGeometryResult,
  cells: CellDetectionResult,
  config: ExtractionConfig
): Promise<ModelResponse<RegionMappingResult>[]> {
  const apiResponses = await callAllModels(imageBase64, prompt, config);
  const results: ModelResponse<RegionMappingResult>[] = [];

  for (const [model, response] of apiResponses) {
    const parsed = response.error ? null : parseRegionResponse(response.content, grid, cells);

    results.push({
      model,
      answer: parsed as RegionMappingResult,
      confidence: parsed?.confidence ?? 0,
      latencyMs: response.latencyMs,
      rawResponse: response.content,
      error: response.error || (parsed === null ? 'Failed to parse response' : undefined),
    });
  }

  return results;
}

function selectBestResult(
  responses: ModelResponse<RegionMappingResult>[],
  grid: GridGeometryResult,
  cells: CellDetectionResult
): RegionMappingResult {
  if (responses.length === 0) {
    // Fallback: single region for all cells
    const shapeLines = cells.shape.split('\n');
    const regions = shapeLines
      .map((line) => line.replace(/\./g, 'A'))
      .join('\n');
    return { regions, confidence: 0 };
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

  // Check for majority vote
  // Note: Region labels can be permuted, so we normalize by structure
  const votes = new Map<string, { count: number; result: RegionMappingResult; totalConfidence: number }>();

  for (const r of responses) {
    // Normalize: count unique regions and their sizes
    const key = normalizeRegionStructure(r.answer.regions);
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

/**
 * Normalize region structure for comparison.
 * Converts region labels to a canonical form based on reading order.
 */
function normalizeRegionStructure(regions: string): string {
  const labelMap = new Map<string, string>();
  let nextLabel = 'A';
  let normalized = '';

  for (const char of regions) {
    if (char === '#' || char === '\n') {
      normalized += char;
    } else {
      if (!labelMap.has(char)) {
        labelMap.set(char, nextLabel);
        nextLabel = String.fromCharCode(nextLabel.charCodeAt(0) + 1);
      }
      normalized += labelMap.get(char);
    }
  }

  return normalized;
}

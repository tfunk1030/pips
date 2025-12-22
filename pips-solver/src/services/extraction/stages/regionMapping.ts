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
  return `Analyze this NYT Pips puzzle screenshot and identify ALL distinct colored regions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRID SHAPE (${grid.rows}×${grid.cols}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

${cells.shape}
(where '.' = cell, '#' = hole)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP-BY-STEP REGION IDENTIFICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SCAN FOR DISTINCT COLORS:
   - Look at each cell's BACKGROUND color (ignore grid lines, numbers, diamonds)
   - Mentally note each unique color you see
   - NYT Pips typically uses 4-8 distinct colors per puzzle

2. IDENTIFY REGION BOUNDARIES:
   - Regions are groups of adjacent cells sharing the SAME color
   - Regions can be any shape (L-shaped, rectangular, irregular)
   - Regions are typically 2-6 cells in size
   - Look for color TRANSITIONS at cell borders

3. ASSIGN LABELS IN READING ORDER:
   - Start at TOP-LEFT corner
   - First new color encountered = Region A
   - Second new color = Region B, etc.
   - Continue left-to-right, top-to-bottom

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NYT PIPS COLOR PALETTE - LEARN TO DISTINGUISH:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WARM COLORS (often confused with each other):
┌──────────────────────────────────────────────────────────────────────┐
│ PINK/CORAL    - Vibrant red-pink, saturated, warm tone              │
│ PEACH/SALMON  - Lighter, more orange-pink, pastel-ish               │
│ ORANGE        - True orange, clearly more yellow than pink          │
│ TAN/BEIGE     - Muted, brownish-yellow, desaturated                 │
└──────────────────────────────────────────────────────────────────────┘

COOL COLORS (often confused with each other):
┌──────────────────────────────────────────────────────────────────────┐
│ TEAL          - Blue-green, darker, more saturated                  │
│ CYAN/AQUA     - Bright blue-green, lighter than teal                │
│ LIGHT BLUE    - Pure blue, no green tint                            │
│ MINT GREEN    - Very pale green with blue undertone                 │
└──────────────────────────────────────────────────────────────────────┘

NEUTRAL & OTHER COLORS:
┌──────────────────────────────────────────────────────────────────────┐
│ GRAY          - Neutral, no color tint                              │
│ OLIVE         - Muddy yellow-green, brownish                        │
│ GREEN         - True green, clearly not olive or teal               │
│ PURPLE/MAUVE  - Pink-purple, clearly different from pink            │
│ YELLOW        - Bright, warm, clearly not tan/beige                 │
└──────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Example 1 - Standard 5x5 puzzle with 6 regions:
Shape:        Colors seen:          Regions:
.....         🟠🟠🔵🔵🟢            AABBC
.....         🟠🟠🔵🔵🟢            AABBC
.....         🟣🟣🟡🟡🟢     →      DDEEC
.....         🟣🟣🟡🟡🟢            DDEEC
.....         🔴🔴🔴🟢🟢            FFFCC

Example 2 - Puzzle with holes (corners cut):
Shape:        Regions:
##...         ##ABB
.....         CCABB
.....    →    CCDDD
.....         EEDDD
...##         EEF##

Example 3 - Similar colors MUST be distinguished:
If you see pink AND coral in same puzzle → They are DIFFERENT regions!
If you see teal AND cyan in same puzzle → They are DIFFERENT regions!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON MISTAKES TO AVOID:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✗ DON'T merge similar colors (pink ≠ coral, teal ≠ cyan, tan ≠ yellow)
✗ DON'T confuse grid lines or shadows with region boundaries
✗ DON'T ignore subtle color differences at boundaries
✗ DON'T assume regions are rectangular - they can be L-shaped or irregular
✗ DON'T count diamonds, numbers, or symbols as separate colors

✓ DO look at the FILL/BACKGROUND color of each cell
✓ DO trace region boundaries by following color changes
✓ DO compare adjacent cells directly to see color differences
✓ DO maintain consistent labeling (same color = same letter everywhere)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT REQUIREMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL RULES:
• Output EXACTLY ${grid.rows} lines with ${grid.cols} characters each
• '#' positions MUST match the shape above exactly
• Every '.' in shape becomes a letter (A-Z) in regions
• Same color = Same letter (no matter where on the grid)
• Different colors = Different letters (even if similar shades)
• Each region typically has 2-6 cells (single-cell regions are rare)

Return ONLY this JSON (no markdown, no explanation, no code blocks):
{"regions": "line1\\nline2\\n...", "confidence": 0.XX}

Confidence scoring:
- 0.95-1.00: All region boundaries crystal clear, colors very distinct
- 0.85-0.94: Very confident, only minor ambiguity at 1-2 boundaries
- 0.70-0.84: Some color boundaries ambiguous, similar colors present
- Below 0.70: Multiple similar colors, several boundaries unclear`;
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

  // Analyze previous attempts to detect potential issues
  const allRegions = previousAttempts.map(a => a.regions);
  const regionCounts = allRegions.map(r => new Set(r.replace(/[#\n]/g, '')).size);
  const avgRegions = regionCounts.length > 0
    ? Math.round(regionCounts.reduce((a, b) => a + b, 0) / regionCounts.length)
    : 0;

  // Check if models are finding different numbers of regions
  const regionCountsVary = regionCounts.length > 1 &&
    Math.max(...regionCounts) - Math.min(...regionCounts) > 0;

  const colorHint = regionCountsVary
    ? `\n⚠️ Models found different numbers of regions (${Math.min(...regionCounts)}-${Math.max(...regionCounts)}).
   This often happens when SIMILAR COLORS are being merged or split incorrectly!`
    : '';

  return `⚠️ RE-EXAMINE: Previous extractions disagreed on region boundaries.

GRID SHAPE (${grid.rows}×${grid.cols}):
${cells.shape}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREVIOUS ATTEMPTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${attemptsStr}
${colorHint}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRECISE COLOR COMPARISON TECHNIQUE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For EACH pair of adjacent cells, ask yourself:

1. SATURATION TEST: Is one cell more "washed out" or more "vibrant"?
   → Different saturation = DIFFERENT regions

2. HUE TEST: Are the colors in different "families"?
   → Pink vs Orange vs Yellow = DIFFERENT regions
   → Teal vs Blue vs Green = DIFFERENT regions

3. BRIGHTNESS TEST: Is one clearly lighter or darker?
   → Light blue vs Dark blue = SAME region (usually)
   → Light pink vs Coral = DIFFERENT regions (check hue too!)

COMMONLY CONFUSED COLOR PAIRS - PAY CLOSE ATTENTION:
┌────────────────────────────────────────────────────────────────────────────┐
│ PINK vs CORAL      - Both warm, but coral has MORE ORANGE                 │
│ PINK vs PEACH      - Peach is LIGHTER and more PASTEL                     │
│ TEAL vs CYAN       - Teal is DARKER and more GREEN-shifted                │
│ TEAL vs BLUE       - Teal has GREEN, blue does not                        │
│ OLIVE vs GREEN     - Olive is MUDDY/BROWN, green is BRIGHT                │
│ TAN vs YELLOW      - Tan is MUTED/BROWN, yellow is BRIGHT                 │
│ GRAY vs BLUE-GRAY  - Blue-gray has slight blue tint                       │
└────────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEMATIC RE-CHECK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Count total distinct colors: "I see __ different colors"
2. For each row, identify colors: "Row 1 has: [color], [color], [color]..."
3. Cross-check adjacent cells: "Does cell (2,3) match cell (2,4)? YES/NO"
4. Build region map one row at a time

NYT puzzles typically have ${avgRegions > 0 ? `around ${avgRegions}` : '4-8'} distinct regions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Exactly ${grid.rows} lines, ${grid.cols} characters each
• '#' must match shape positions exactly
• Only A-Z letters for cells (no numbers, no '.')
• Each region needs at least 2 cells (single-cell regions are very rare)
• Same color ANYWHERE = Same letter (consistency is key!)

Return ONLY this JSON (no markdown, no explanation):
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

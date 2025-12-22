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

  return `Analyze this NYT Pips puzzle screenshot and count the EXACT pips on each domino in the tray.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED DOMINO COUNT: ${expectedDominoes} dominoes (based on ${cellCount} grid cells)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP-BY-STEP COUNTING METHOD:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LOCATE THE DOMINO TRAY:
   - Find the reference area showing available dominoes (usually at BOTTOM of screen)
   - Dominoes are rectangular tiles split into two halves
   - Each half contains 0-6 pips (dots)

2. COUNT DOMINOES LEFT-TO-RIGHT:
   - Start from the LEFT-MOST domino
   - Move RIGHT through each domino in sequence
   - Make sure you count ALL ${expectedDominoes} dominoes

3. FOR EACH DOMINO, COUNT BOTH HALVES:
   - Look at the LEFT half first, count pips
   - Look at the RIGHT half, count pips
   - Record as [left_pips, right_pips]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIP PATTERN VISUAL REFERENCE (memorize these patterns!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0 PIPS (blank):     1 PIP (center):     2 PIPS (diagonal):
┌─────────┐         ┌─────────┐         ┌─────────┐
│         │         │         │         │ ●       │
│         │         │    ●    │         │         │
│         │         │         │         │       ● │
└─────────┘         └─────────┘         └─────────┘

3 PIPS (diagonal):  4 PIPS (corners):   5 PIPS (corners+center):
┌─────────┐         ┌─────────┐         ┌─────────┐
│ ●       │         │ ●     ● │         │ ●     ● │
│    ●    │         │         │         │    ●    │
│       ● │         │ ●     ● │         │ ●     ● │
└─────────┘         └─────────┘         └─────────┘

6 PIPS (two columns of 3):
┌─────────┐
│ ●     ● │
│ ●     ● │
│ ●     ● │
└─────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DISTINGUISHING FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ 5 vs 6 (MOST COMMON MISTAKE):
   - 5: Has CENTER pip + 4 corners (forms an X pattern)
   - 6: NO center pip, just 2 vertical columns of 3 (forms || pattern)

⚠️ 3 vs 5:
   - 3: Only 3 pips in a diagonal line (top-left to bottom-right)
   - 5: 5 pips - 4 corners PLUS center dot

⚠️ 2 vs 4:
   - 2: Only 2 pips on opposite corners (diagonal)
   - 4: 4 pips in ALL four corners

⚠️ 0 vs hard-to-see:
   - 0 (blank): Completely empty half, no dots at all
   - If you see ANY dot, it's not zero

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON MISTAKES TO AVOID:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✗ DON'T confuse 5 and 6 - check for the CENTER dot!
✗ DON'T miss blank (0) halves - they look completely empty
✗ DON'T skip dominoes at the edges of the tray
✗ DON'T count dominoes from the puzzle grid (only the tray!)
✗ DON'T report duplicate dominoes (each domino is unique in NYT Pips)

✓ DO count pips by looking for the CENTER first (odd numbers have center dots)
✓ DO verify your total matches ${expectedDominoes} dominoes
✓ DO check corners first for even-numbered pip counts (2, 4, 6)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERIFICATION CHECKLIST:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before responding, verify:
□ Found exactly ${expectedDominoes} dominoes
□ Each domino has two values between 0-6
□ No duplicate dominoes in your list
□ Double-checked any 5s and 6s (center dot test)
□ Double-checked any 0s (truly blank, not obscured)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY this JSON (no markdown, no explanation, no code blocks):
{"dominoes": [[left1, right1], [left2, right2], ...], "confidence": 0.XX}

Confidence scoring:
- 0.95-1.00: All pips crystal clear, 100% certain of all counts
- 0.85-0.94: Very confident, maybe 1 domino slightly unclear
- 0.70-0.84: Moderately confident, some pips hard to distinguish
- Below 0.70: Low confidence, multiple unclear pip counts`;
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
    .map((a, i) => `Model ${i + 1}: ${JSON.stringify(a.dominoes)} (${a.dominoes.length} dominoes)`)
    .join('\n');

  // Find disagreements between attempts for targeted guidance
  const allDominoes = previousAttempts.flatMap(a => a.dominoes);
  const has5or6 = allDominoes.some(([p1, p2]) => p1 >= 5 || p2 >= 5);
  const has0 = allDominoes.some(([p1, p2]) => p1 === 0 || p2 === 0);

  return `⚠️ RE-EXAMINE: Previous extractions disagreed on domino pip counts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED: ${expectedDominoes} dominoes (for ${cellCount} cells)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREVIOUS RESULTS THAT DISAGREED:
${attemptsStr}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRECISE COUNTING TECHNIQUE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LOCATE THE DOMINO TRAY (bottom of screenshot, NOT the puzzle grid)

2. COUNT EACH DOMINO ONE-BY-ONE:
   For each domino, say to yourself:
   "Domino 1: left half has ___ pips, right half has ___ pips"
   "Domino 2: left half has ___ pips, right half has ___ pips"
   ... and so on

3. USE THE CENTER DOT TEST:
   - Look at the CENTER of the half first
   - If there's a center dot: could be 1, 3, or 5
   - If NO center dot: could be 0, 2, 4, or 6

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PIP PATTERN QUICK REFERENCE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WITH CENTER DOT (odd):        WITHOUT CENTER DOT (even):
• 1 = center only             • 0 = blank/empty
• 3 = diagonal line           • 2 = two diagonal corners
• 5 = X pattern (4+center)    • 4 = four corners
                              • 6 = two columns (|| pattern)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${has5or6 ? `⚠️ SPECIAL ATTENTION - 5 vs 6 DETECTED IN PREVIOUS ATTEMPTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5 PIPS:                    6 PIPS:
┌─────────┐                ┌─────────┐
│ ●     ● │                │ ●     ● │
│    ●    │  ← HAS CENTER  │ ●     ● │  ← NO CENTER
│ ●     ● │                │ ●     ● │
└─────────┘                └─────────┘

KEY TEST: Is there a dot in the EXACT CENTER?
  YES → It's 5 (X pattern)
  NO  → It's 6 (|| pattern)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
` : ''}${has0 ? `⚠️ SPECIAL ATTENTION - BLANK (0) DETECTED IN PREVIOUS ATTEMPTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0 PIPS (blank) means COMPLETELY EMPTY - no dots at all.
If you see even ONE dot, it's not zero!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
` : ''}CRITICAL REMINDERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• ONLY look at the domino tray (NOT the puzzle grid)
• You must find EXACTLY ${expectedDominoes} dominoes
• Each domino in NYT Pips is UNIQUE (no duplicate pairs)
• Check EVERY domino, including ones at the edges

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY this JSON (no markdown, no explanation):
{"dominoes": [[left1, right1], [left2, right2], ...], "confidence": 0.XX}`;
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

  if (validResponses.length < 2 && retryCount < config.maxRetries) {
    retryCount++;
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

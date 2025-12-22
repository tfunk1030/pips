/**
 * Stage 4: Constraint Extraction
 *
 * Extracts region constraints (sum, all_equal) from the puzzle.
 * Uses 3-model ensemble with consensus voting.
 */

import {
  ExtractionConfig,
  Constraint,
  ConstraintExtractionResult,
  ModelResponse,
  RegionMappingResult,
} from '../types';
import { callAllModels } from '../apiClient';
import { NYT_VALIDATION } from '../config';

// =============================================================================
// Prompt Template
// =============================================================================

function getConstraintPrompt(regions: RegionMappingResult): string {
  // Extract unique region labels
  const labels = new Set<string>();
  for (const char of regions.regions) {
    if (char !== '#' && char !== '\n') {
      labels.add(char);
    }
  }
  const regionLabels = Array.from(labels).sort().join(', ');

  return `Analyze this NYT Pips puzzle and extract the constraints for each colored region.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGIONS TO FIND CONSTRAINTS FOR: ${regionLabels}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REGIONS MAP (for reference):
${regions.regions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIAMOND LABEL IDENTIFICATION GUIDE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NYT Pips displays constraints as SMALL WHITE DIAMOND SHAPES (◇) within cells:

DIAMOND APPEARANCE:
  ┌─────────────────┐
  │   ╱╲            │  ← The diamond is a small
  │  ╱12╲           │     rotated square shape
  │  ╲  ╱           │     containing a number
  │   ╲╱            │     or symbol inside
  └─────────────────┘

WHERE TO LOOK:
- Diamonds appear INSIDE one cell of each colored region
- Usually positioned in the CENTER of the cell
- The diamond is WHITE/light-colored against the cell's background
- Look carefully - they can be small (about 1/3 of cell size)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONSTRAINT TYPES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SUM CONSTRAINTS (most common):
   ┌──────────┐
   │  ◇ 12    │  → Sum of all domino pips in region = 12
   └──────────┘
   - A NUMBER inside the diamond (e.g., 5, 8, 12, 15, 21)
   - Default operator is "==" (sum equals the value)
   - Possible operators: "==" (equals), "<" (less than), ">" (greater than)

   VISUAL EXAMPLES OF SUM VALUES:
   ◇5   ◇8   ◇10  ◇12  ◇15  ◇18  ◇21

2. ALL_EQUAL CONSTRAINTS:
   ┌──────────┐
   │  ◇ =     │  → All domino pips in region must be same value
   └──────────┘
   - Shows "=" symbol OR letter "E" inside diamond
   - No number value needed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP-BY-STEP READING PROCESS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For EACH colored region (${regionLabels}):

1. LOCATE THE REGION in the puzzle by its color
2. SCAN ALL CELLS in that region for a white diamond shape (◇)
3. ZOOM IN mentally on the diamond to read its contents
4. IDENTIFY what's inside:
   - If it's a NUMBER → SUM constraint with op "=="
   - If it's "=" or "E" → ALL_EQUAL constraint
   - If there's "<" before number → SUM with op "<"
   - If there's ">" before number → SUM with op ">"
5. RECORD the constraint for this region

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMONLY CONFUSED CHARACTERS - READ CAREFULLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NUMBER PAIRS TO DISTINGUISH:
  • 6 vs 8  → 6 has one loop (top), 8 has two stacked loops
  • 3 vs 8  → 3 is open on the left, 8 is fully closed
  • 1 vs 7  → 1 is straight vertical, 7 has a horizontal top
  • 5 vs 6  → 5 has flat top, 6 has curved top
  • 9 vs 6  → 9 has loop at top, 6 has loop at bottom
  • 10 vs 16 → Count digits: 10 is "1" + "0", 16 is "1" + "6"
  • 11 vs 17 → 11 is two "1"s, 17 has "7" on right
  • 12 vs 18 → Check second digit carefully: 2 vs 8

OPERATOR SYMBOLS:
  • "=" → Equals sign, two horizontal lines (ALL_EQUAL)
  • "<" → Less than, pointing LEFT (open side on right)
  • ">" → Greater than, pointing RIGHT (open side on left)
  • "E" → Letter E for Equal (same as =)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMON MISTAKES TO AVOID:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✗ DON'T confuse domino pips (dots) with constraint diamonds
✗ DON'T assume constraints are always in center cell - check ALL cells in region
✗ DON'T misread 6 as 8 or 8 as 6 (very common error!)
✗ DON'T skip regions - every region should have exactly one constraint
✗ DON'T add operators where there are none (default is "==")
✗ DON'T invent constraints you can't clearly see

✓ DO check each colored region systematically
✓ DO read each digit individually for multi-digit numbers (1-2 vs 12)
✓ DO distinguish between "=" symbol (all_equal) and a number
✓ DO verify your reading by asking "does this value make sense?"
   (Typical sum values: 5-25, rarely above 30)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED VALUE RANGES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Sum values typically range from 0 to 42
- Small regions (2-3 cells): expect sums 2-18
- Medium regions (4-5 cells): expect sums 4-30
- Large regions (6+ cells): expect sums 6-42
- All_equal constraints have no value (just type)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY this JSON (no markdown, no explanation, no code blocks):
{
  "constraints": {
    "A": {"type": "sum", "op": "==", "value": 12},
    "B": {"type": "all_equal"},
    "C": {"type": "sum", "op": "<", "value": 8}
  },
  "confidence": 0.XX
}

VALID VALUES:
- type: "sum" or "all_equal"
- op (for sum only): "==" (default), "<", ">"
- value (for sum only): 0-42

Confidence scoring:
- 0.95-1.00: All diamonds clearly visible, numbers easy to read
- 0.85-0.94: Most constraints clear, one or two slightly ambiguous
- 0.70-0.84: Some diamonds hard to read, uncertain about digits
- Below 0.70: Multiple constraints unclear or potentially misread`;
}

// =============================================================================
// Retry Prompt
// =============================================================================

function getRetryPrompt(
  regions: RegionMappingResult,
  previousAttempts: ConstraintExtractionResult[]
): string {
  const attemptsStr = previousAttempts
    .map((a, i) => `Model ${i + 1}: ${JSON.stringify(a.constraints)}`)
    .join('\n');

  // Analyze disagreements to provide targeted hints
  const allConstraints: Record<string, Set<string>> = {};
  for (const attempt of previousAttempts) {
    for (const [region, constraint] of Object.entries(attempt.constraints)) {
      if (!allConstraints[region]) {
        allConstraints[region] = new Set();
      }
      const key = constraint.type === 'sum'
        ? `sum(${constraint.op || '=='},${constraint.value})`
        : 'all_equal';
      allConstraints[region].add(key);
    }
  }

  // Find regions with disagreements
  const disagreedRegions: string[] = [];
  const valueDisagreements: string[] = [];
  for (const [region, values] of Object.entries(allConstraints)) {
    if (values.size > 1) {
      disagreedRegions.push(region);
      const valuesArr = Array.from(values);
      // Check for common confusion patterns (6 vs 8, etc.)
      const nums = valuesArr
        .map(v => {
          const match = v.match(/sum\([^,]+,(\d+)\)/);
          return match ? parseInt(match[1]) : null;
        })
        .filter(n => n !== null) as number[];

      if (nums.length >= 2) {
        const diff = Math.abs(nums[0] - nums[1]);
        if (diff === 2 && (nums.includes(6) || nums.includes(8))) {
          valueDisagreements.push(`⚠️ Region ${region}: Disagreement between 6 and 8 - VERY common confusion!`);
        } else if (diff === 6 && (nums.includes(10) || nums.includes(16))) {
          valueDisagreements.push(`⚠️ Region ${region}: Disagreement between 10 and 16 - check second digit carefully`);
        } else if (diff === 6 && (nums.includes(12) || nums.includes(18))) {
          valueDisagreements.push(`⚠️ Region ${region}: Disagreement between 12 and 18 - check second digit (2 vs 8)`);
        }
      }
    }
  }

  const disagreementHints = valueDisagreements.length > 0
    ? `\n${valueDisagreements.join('\n')}`
    : '';

  const focusRegions = disagreedRegions.length > 0
    ? `\n🎯 FOCUS ON REGIONS: ${disagreedRegions.join(', ')} (these had disagreements)`
    : '';

  return `⚠️ RE-EXAMINE: Previous constraint extractions disagreed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PREVIOUS RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${attemptsStr}${focusRegions}${disagreementHints}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REGIONS MAP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${regions.regions}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRECISE DIAMOND READING TECHNIQUE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For each region, follow this EXACT process:

1. FIND THE DIAMOND:
   - Scan every cell in the colored region
   - Look for a small WHITE rotated square (◇) shape
   - It will be in the CENTER of one cell

2. READ THE CONTENT CAREFULLY:

   For NUMBERS:
   ┌────────────────────────────────────────────────────────────┐
   │ DIGIT VERIFICATION CHECKLIST:                              │
   │                                                            │
   │ • Does it have TWO loops stacked? → 8                      │
   │ • Does it have ONE loop with open bottom? → 6              │
   │ • Does it have ONE loop with open top? → 9                 │
   │ • Is it a curved 'S' shape open on left? → 3               │
   │ • Is it two vertical lines? → 11                           │
   │ • First digit "1" + round "0"? → 10                        │
   │ • First digit "1" + curved "2"? → 12                       │
   │ • First digit "1" + two-loop "8"? → 18                     │
   └────────────────────────────────────────────────────────────┘

   For SYMBOLS:
   • Two horizontal lines (=) → all_equal
   • Letter "E" → all_equal
   • Left-pointing angle (<) → sum with op "<"
   • Right-pointing angle (>) → sum with op ">"

3. VERIFY YOUR READING:
   - Re-examine the digit shape one more time
   - Ask: "Am I 100% certain this is a 6 and not an 8?"
   - Ask: "Is this a two-digit number or single digit?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL REMINDERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• 6 vs 8 is the MOST COMMON misread - 8 has two loops, 6 has one
• Multi-digit numbers: read each digit separately (1+2=12, not just "12")
• "=" symbol means all_equal, NOT a sum of 0
• Every region typically has exactly one constraint
• The default operator for sum is "==" (not "<" or ">")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY this JSON (no markdown, no explanation):
{
  "constraints": {"A": {"type": "sum", "op": "==", "value": N}, ...},
  "confidence": 0.XX
}`;
}

// =============================================================================
// Response Parser
// =============================================================================

function parseConstraintResponse(
  content: string,
  regions: RegionMappingResult
): ConstraintExtractionResult | null {
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

    if (typeof parsed.constraints !== 'object' || typeof parsed.confidence !== 'number') {
      return null;
    }

    // Validate and normalize constraints
    const constraints: Record<string, Constraint> = {};

    for (const [region, constraint] of Object.entries(parsed.constraints)) {
      if (typeof constraint !== 'object' || constraint === null) {
        continue;
      }

      const c = constraint as Record<string, unknown>;

      if (c.type === 'sum') {
        if (
          typeof c.op === 'string' &&
          ['==', '<', '>'].includes(c.op) &&
          typeof c.value === 'number'
        ) {
          constraints[region] = {
            type: 'sum',
            op: c.op as '==' | '<' | '>',
            value: Math.round(c.value),
          };
        }
      } else if (c.type === 'all_equal') {
        constraints[region] = { type: 'all_equal' };
      }
    }

    return {
      constraints,
      confidence: Math.min(1, Math.max(0, parsed.confidence)),
    };
  } catch {
    return null;
  }
}

// =============================================================================
// Validation
// =============================================================================

function validateConstraintResult(
  result: ConstraintExtractionResult,
  regions: RegionMappingResult
): string[] {
  const errors: string[] = [];

  // Get valid region labels
  const validLabels = new Set<string>();
  for (const char of regions.regions) {
    if (char !== '#' && char !== '\n') {
      validLabels.add(char);
    }
  }

  for (const [region, constraint] of Object.entries(result.constraints)) {
    // Check region exists
    if (!validLabels.has(region)) {
      errors.push(`Constraint for unknown region "${region}"`);
      continue;
    }

    // Validate sum constraints
    if (constraint.type === 'sum') {
      if (constraint.value === undefined) {
        errors.push(`Sum constraint for ${region} missing value`);
      } else if (constraint.value < 0 || constraint.value > NYT_VALIDATION.maxSumValue) {
        errors.push(`Sum value ${constraint.value} for ${region} outside valid range [0, ${NYT_VALIDATION.maxSumValue}]`);
      }

      if (!constraint.op || !['==', '<', '>'].includes(constraint.op)) {
        errors.push(`Sum constraint for ${region} has invalid operator: ${constraint.op}`);
      }
    }
  }

  return errors;
}

// =============================================================================
// Main Extraction Function
// =============================================================================

export interface ConstraintStageResult {
  result: ConstraintExtractionResult;
  responses: ModelResponse<ConstraintExtractionResult>[];
  retryCount: number;
  validationErrors: string[];
}

/**
 * Extract constraints from image using 3-model ensemble
 */
export async function extractConstraints(
  imageBase64: string,
  regions: RegionMappingResult,
  config: ExtractionConfig
): Promise<ConstraintStageResult> {
  const allResponses: ModelResponse<ConstraintExtractionResult>[] = [];
  let retryCount = 0;

  // Initial extraction
  const prompt = getConstraintPrompt(regions);
  let responses = await callModelsForConstraints(imageBase64, prompt, regions, config);
  allResponses.push(...responses);

  // Check if we need to retry
  let validResponses = responses.filter((r) => r.answer !== null && !r.error);

  if (validResponses.length < 2 && retryCount < config.maxRetries) {
    retryCount++;
    const retryResponses = await callModelsForConstraints(
      imageBase64,
      getRetryPrompt(regions, validResponses.map((r) => r.answer!)),
      regions,
      config
    );
    allResponses.push(...retryResponses);
    validResponses = [
      ...validResponses,
      ...retryResponses.filter((r) => r.answer !== null && !r.error),
    ];
  }

  // Select best result using consensus
  const result = selectBestResult(validResponses as ModelResponse<ConstraintExtractionResult>[]);
  const validationErrors = validateConstraintResult(result, regions);

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

async function callModelsForConstraints(
  imageBase64: string,
  prompt: string,
  regions: RegionMappingResult,
  config: ExtractionConfig
): Promise<ModelResponse<ConstraintExtractionResult>[]> {
  const apiResponses = await callAllModels(imageBase64, prompt, config);
  const results: ModelResponse<ConstraintExtractionResult>[] = [];

  for (const [model, response] of apiResponses) {
    const parsed = response.error ? null : parseConstraintResponse(response.content, regions);

    results.push({
      model,
      answer: parsed as ConstraintExtractionResult,
      confidence: parsed?.confidence ?? 0,
      latencyMs: response.latencyMs,
      rawResponse: response.content,
      error: response.error || (parsed === null ? 'Failed to parse response' : undefined),
    });
  }

  return results;
}

function selectBestResult(
  responses: ModelResponse<ConstraintExtractionResult>[]
): ConstraintExtractionResult {
  if (responses.length === 0) {
    return { constraints: {}, confidence: 0 };
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

  // For constraints, we do per-region voting
  const mergedConstraints: Record<string, Constraint> = {};
  const allRegions = new Set<string>();

  for (const r of responses) {
    for (const region of Object.keys(r.answer.constraints)) {
      allRegions.add(region);
    }
  }

  for (const region of allRegions) {
    const votes = new Map<string, { count: number; constraint: Constraint }>();

    for (const r of responses) {
      const constraint = r.answer.constraints[region];
      if (constraint) {
        const key = JSON.stringify(constraint);
        const existing = votes.get(key);
        if (existing) {
          existing.count++;
        } else {
          votes.set(key, { count: 1, constraint });
        }
      }
    }

    // Find majority or use highest frequency
    let bestVote: { count: number; constraint: Constraint } | null = null;
    for (const vote of votes.values()) {
      if (!bestVote || vote.count > bestVote.count) {
        bestVote = vote;
      }
    }

    if (bestVote && bestVote.count >= 2) {
      mergedConstraints[region] = bestVote.constraint;
    } else if (bestVote) {
      // No majority, use highest confidence model's answer
      const topConstraint = top.answer.constraints[region];
      if (topConstraint) {
        mergedConstraints[region] = topConstraint;
      }
    }
  }

  return {
    constraints: mergedConstraints,
    confidence: (top.confidence + second.confidence) / 2,
  };
}

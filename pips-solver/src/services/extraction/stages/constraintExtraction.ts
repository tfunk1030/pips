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

  return `Given this NYT Pips puzzle, extract the constraints for each region.

REGIONS FOUND: ${regionLabels}

REGIONS MAP:
${regions.regions}

CONSTRAINT TYPES IN NYT PIPS:
1. SUM constraints - diamond shape with a number (e.g., ◇12)
   - "==" means sum must equal the value
   - "<" means sum must be less than the value
   - ">" means sum must be greater than the value

2. ALL_EQUAL constraints - often shown as "=" or "E"
   - All pip values in the region must be the same

WHERE TO FIND CONSTRAINTS:
- Look for diamond shapes (◇) near or inside each colored region
- Numbers inside diamonds indicate sum constraints
- The "=" symbol or "E" indicates all_equal

INSTRUCTIONS:
1. For each region (${regionLabels}), find its constraint
2. Extract the type, operator (for sum), and value (for sum)
3. Not all regions may have visible constraints - only include what you see

Return ONLY valid JSON (no markdown, no explanation):
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
- op (for sum only): "==", "<", ">"
- value (for sum only): 0-42 (max possible sum is 7 cells × 6 pips)

Confidence scoring:
- 0.95-1.00: All constraint symbols clearly visible
- 0.85-0.94: Most constraints clear, minor ambiguity
- 0.70-0.84: Some constraints hard to read
- Below 0.70: Multiple constraints unclear`;
}

// =============================================================================
// Retry Prompt
// =============================================================================

function getRetryPrompt(
  regions: RegionMappingResult,
  previousAttempts: ConstraintExtractionResult[]
): string {
  const attemptsStr = previousAttempts
    .map((a, i) => `Attempt ${i + 1}: ${JSON.stringify(a.constraints)}`)
    .join('\n');

  return `Extract constraints for each region in this NYT Pips puzzle.

REGIONS MAP:
${regions.regions}

PREVIOUS ATTEMPTS DISAGREED:
${attemptsStr}

Please look again MORE CAREFULLY:
1. Diamond symbols (◇) contain constraint numbers
2. Look INSIDE each colored region for its constraint marker
3. "=" or "E" means all_equal (all pips same value)
4. Numbers like 12, 8, 15 are sum target values

COMMON MISTAKES TO AVOID:
- Confusing similar numbers (6 vs 8, 3 vs 8)
- Missing small constraint markers
- Misreading operator (< vs > vs =)

Return ONLY valid JSON:
{
  "constraints": {"A": {...}, "B": {...}},
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

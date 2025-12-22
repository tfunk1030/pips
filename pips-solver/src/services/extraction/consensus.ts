/**
 * Consensus Algorithm
 *
 * Implements confidence-weighted consensus with majority fallback
 * for multi-model ensemble results.
 */

import { ModelResponse, ConsensusResult } from './types';

// =============================================================================
// Generic Consensus Function
// =============================================================================

export interface ConsensusOptions<T> {
  /** Function to convert answer to comparable key */
  keyFn?: (answer: T) => string;
  /** Confidence difference threshold for using highest confidence (default: 0.10) */
  confidenceThreshold?: number;
}

/**
 * Resolve consensus from multiple model responses.
 *
 * Algorithm:
 * 1. If top confidence is significantly higher (>threshold), use it
 * 2. Otherwise, use majority vote
 * 3. If no majority, use highest confidence (best-effort)
 */
export function resolveConsensus<T>(
  responses: ModelResponse<T>[],
  options: ConsensusOptions<T> = {}
): ConsensusResult<T> {
  const { keyFn = JSON.stringify, confidenceThreshold = 0.10 } = options;

  // Filter valid responses
  const validResponses = responses.filter(
    (r) => r.answer !== null && r.answer !== undefined && !r.error
  );

  if (validResponses.length === 0) {
    throw new Error('No valid responses to resolve consensus');
  }

  // Sort by confidence (descending)
  const sorted = [...validResponses].sort((a, b) => b.confidence - a.confidence);
  const top = sorted[0];

  if (validResponses.length === 1) {
    return {
      answer: top.answer,
      source: 'confidence',
      confident: top.confidence >= 0.7,
      responses: validResponses,
      retryCount: 0,
    };
  }

  const second = sorted[1];

  // Step 1: Check for high-confidence winner
  if (top.confidence - second.confidence > confidenceThreshold) {
    return {
      answer: top.answer,
      source: 'confidence',
      confident: true,
      responses: validResponses,
      retryCount: 0,
    };
  }

  // Step 2: Use majority vote
  const votes = new Map<string, { count: number; answer: T; totalConfidence: number }>();

  for (const r of validResponses) {
    const key = keyFn(r.answer);
    const existing = votes.get(key);
    if (existing) {
      existing.count++;
      existing.totalConfidence += r.confidence;
    } else {
      votes.set(key, { count: 1, answer: r.answer, totalConfidence: r.confidence });
    }
  }

  // Find majority (2+ votes for 3 models)
  for (const vote of votes.values()) {
    if (vote.count >= 2) {
      return {
        answer: vote.answer,
        source: 'majority',
        confident: true,
        responses: validResponses,
        retryCount: 0,
      };
    }
  }

  // Step 3: No majority, use highest confidence (best-effort)
  return {
    answer: top.answer,
    source: 'best-effort',
    confident: false,
    responses: validResponses,
    retryCount: 0,
  };
}

// =============================================================================
// Specialized Consensus Functions
// =============================================================================

/**
 * Consensus for grid dimensions [rows, cols]
 */
export function resolveGridConsensus<T extends { rows: number; cols: number }>(
  responses: ModelResponse<T>[]
): ConsensusResult<T> {
  return resolveConsensus(responses, {
    keyFn: (answer) => `${answer.rows}x${answer.cols}`,
  });
}

/**
 * Consensus for shape/regions strings (normalized)
 */
export function resolveShapeConsensus<T extends { shape?: string; regions?: string }>(
  responses: ModelResponse<T>[]
): ConsensusResult<T> {
  return resolveConsensus(responses, {
    keyFn: (answer) => {
      const str = answer.shape || answer.regions || '';
      return str.replace(/\s+/g, ''); // Normalize whitespace
    },
  });
}

/**
 * Consensus for dominoes (order-independent)
 */
export function resolveDominoConsensus<T extends { dominoes: [number, number][] }>(
  responses: ModelResponse<T>[]
): ConsensusResult<T> {
  return resolveConsensus(responses, {
    keyFn: (answer) => {
      // Normalize: sort each domino [min, max], then sort the list
      const normalized = answer.dominoes
        .map(([a, b]) => `${Math.min(a, b)}-${Math.max(a, b)}`)
        .sort()
        .join(',');
      return normalized;
    },
  });
}

// =============================================================================
// Retry Decision Logic
// =============================================================================

export interface RetryDecision {
  shouldRetry: boolean;
  reason: string;
}

/**
 * Determine if a stage should retry based on consensus result
 */
export function shouldRetry<T>(
  consensus: ConsensusResult<T>,
  validationErrors: string[],
  currentRetryCount: number,
  maxRetries: number
): RetryDecision {
  // Already at max retries
  if (currentRetryCount >= maxRetries) {
    return { shouldRetry: false, reason: 'Max retries reached' };
  }

  // Validation failed
  if (validationErrors.length > 0) {
    return {
      shouldRetry: true,
      reason: `Validation errors: ${validationErrors.join('; ')}`,
    };
  }

  // No confident consensus
  if (!consensus.confident) {
    return {
      shouldRetry: true,
      reason: 'No confident consensus reached',
    };
  }

  // Less than 2 valid responses
  if (consensus.responses.length < 2) {
    return {
      shouldRetry: true,
      reason: 'Insufficient valid responses',
    };
  }

  return { shouldRetry: false, reason: 'Consensus reached' };
}

// =============================================================================
// Merge Strategies for Partial Consensus
// =============================================================================

/**
 * Merge constraint results from multiple responses.
 * Uses per-region voting for partial consensus.
 */
export function mergeConstraints(
  responses: ModelResponse<{ constraints: Record<string, unknown> }>[]
): Record<string, unknown> {
  const allRegions = new Set<string>();
  const regionVotes = new Map<string, Map<string, { count: number; value: unknown }>>();

  // Collect all regions and their values
  for (const r of responses) {
    if (!r.answer || r.error) continue;

    for (const [region, constraint] of Object.entries(r.answer.constraints)) {
      allRegions.add(region);

      if (!regionVotes.has(region)) {
        regionVotes.set(region, new Map());
      }

      const key = JSON.stringify(constraint);
      const votes = regionVotes.get(region)!;
      const existing = votes.get(key);

      if (existing) {
        existing.count++;
      } else {
        votes.set(key, { count: 1, value: constraint });
      }
    }
  }

  // Select best constraint per region
  const merged: Record<string, unknown> = {};

  for (const region of allRegions) {
    const votes = regionVotes.get(region);
    if (!votes) continue;

    let best: { count: number; value: unknown } | null = null;
    for (const vote of votes.values()) {
      if (!best || vote.count > best.count) {
        best = vote;
      }
    }

    if (best) {
      merged[region] = best.value;
    }
  }

  return merged;
}

/**
 * Merge domino results from multiple responses.
 * Uses per-domino voting for partial consensus.
 */
export function mergeDominoes(
  responses: ModelResponse<{ dominoes: [number, number][] }>[],
  expectedCount: number
): [number, number][] {
  const dominoVotes = new Map<string, { count: number; domino: [number, number] }>();

  // Collect votes for each domino
  for (const r of responses) {
    if (!r.answer || r.error) continue;

    for (const [p1, p2] of r.answer.dominoes) {
      const key = `${Math.min(p1, p2)}-${Math.max(p1, p2)}`;
      const existing = dominoVotes.get(key);

      if (existing) {
        existing.count++;
      } else {
        dominoVotes.set(key, {
          count: 1,
          domino: [Math.min(p1, p2), Math.max(p1, p2)] as [number, number],
        });
      }
    }
  }

  // Sort by vote count (descending), take top expectedCount
  const sorted = Array.from(dominoVotes.values()).sort((a, b) => b.count - a.count);

  return sorted.slice(0, expectedCount).map((v) => v.domino);
}

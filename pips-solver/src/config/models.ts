/**
 * Model Configuration
 * Claude model IDs for AI extraction features
 */

// Prefer newest supported models, but gracefully fall back if the account
// doesn't have access to a given model ID.
export const MODEL_CANDIDATES = [
  'claude-sonnet-4-20250514',
  // Older fallbacks (may be retired for some accounts, but cheap to try)
  'claude-3-5-sonnet-20240620',
  'claude-3-opus-20240229',
] as const;

export type ModelId = (typeof MODEL_CANDIDATES)[number];
export const DEFAULT_MODEL = MODEL_CANDIDATES[0];

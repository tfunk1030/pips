/**
 * Multi-Stage Extraction Pipeline Configuration
 *
 * Default configuration and factory functions for the extraction pipeline.
 */

import { ExtractionConfig } from './types';

// =============================================================================
// Model Identifiers (OpenRouter format)
// =============================================================================

export const DEFAULT_MODELS = {
  // Model IDs as of December 2025 - these exist on OpenRouter
  gemini: 'google/gemini-3-pro-preview',
  gpt: 'openai/gpt-5.2',
  claude: 'anthropic/claude-opus-4.5',
} as const;

// =============================================================================
// NYT Pips Validation Bounds
// =============================================================================

export const NYT_VALIDATION = {
  /** Minimum grid dimension (rows or cols) */
  minGridSize: 3,
  /** Maximum grid dimension (rows or cols) */
  maxGridSize: 8,
  /** Valid pip range [min, max] */
  pipRange: [0, 6] as [number, number],
  /** NYT uses unique dominoes (no duplicates) */
  uniqueDominoes: true,
  /** Minimum cells for a valid puzzle */
  minCells: 8,
  /** Maximum sum constraint value (7 cells × 6 pips) */
  maxSumValue: 42,
} as const;

// =============================================================================
// Default Configuration
// =============================================================================

export const DEFAULT_CONFIG: ExtractionConfig = {
  apiKeys: {},
  models: { ...DEFAULT_MODELS },
  maxRetries: 2,
  confidenceThreshold: 0.10,
  timeoutMs: 30000,
  validation: {
    minGridSize: NYT_VALIDATION.minGridSize,
    maxGridSize: NYT_VALIDATION.maxGridSize,
    pipRange: NYT_VALIDATION.pipRange,
    uniqueDominoes: NYT_VALIDATION.uniqueDominoes,
  },
  lowConfidenceThreshold: 0.70,
  saveDebugResponses: false,
  enableLegacyFallback: true,
};

// =============================================================================
// Configuration Factory
// =============================================================================

/**
 * Create extraction config with overrides
 */
export function createConfig(overrides: Partial<ExtractionConfig> = {}): ExtractionConfig {
  return {
    ...DEFAULT_CONFIG,
    ...overrides,
    apiKeys: {
      ...DEFAULT_CONFIG.apiKeys,
      ...overrides.apiKeys,
    },
    models: {
      ...DEFAULT_CONFIG.models,
      ...overrides.models,
    },
    validation: {
      ...DEFAULT_CONFIG.validation,
      ...overrides.validation,
    },
  };
}

/**
 * Create config with OpenRouter API key
 */
export function createOpenRouterConfig(apiKey: string, overrides: Partial<ExtractionConfig> = {}): ExtractionConfig {
  return createConfig({
    ...overrides,
    apiKeys: {
      openrouter: apiKey,
      ...overrides.apiKeys,
    },
  });
}

/**
 * Create config with individual provider API keys
 */
export function createDirectConfig(
  keys: { google?: string; openai?: string; anthropic?: string },
  overrides: Partial<ExtractionConfig> = {}
): ExtractionConfig {
  return createConfig({
    ...overrides,
    apiKeys: {
      google: keys.google,
      openai: keys.openai,
      anthropic: keys.anthropic,
      ...overrides.apiKeys,
    },
  });
}

// =============================================================================
// Validation Helpers
// =============================================================================

/**
 * Check if config has at least one valid API key configuration
 */
export function hasValidApiKeys(config: ExtractionConfig): boolean {
  const { apiKeys } = config;

  // OpenRouter covers all models
  if (apiKeys.openrouter) {
    return true;
  }

  // Need at least one provider key for ensemble
  const hasGoogle = !!apiKeys.google;
  const hasOpenAI = !!apiKeys.openai;
  const hasAnthropic = !!apiKeys.anthropic;

  // For best results, need all 3, but can work with at least 1
  return hasGoogle || hasOpenAI || hasAnthropic;
}

/**
 * Get list of available models based on API keys
 */
export function getAvailableModels(config: ExtractionConfig): string[] {
  const { apiKeys, models } = config;
  const available: string[] = [];

  if (apiKeys.openrouter) {
    // OpenRouter can access all models
    available.push(models.gemini, models.gpt, models.claude);
  } else {
    if (apiKeys.google) available.push(models.gemini);
    if (apiKeys.openai) available.push(models.gpt);
    if (apiKeys.anthropic) available.push(models.claude);
  }

  return available;
}

/**
 * Check if we have enough models for ensemble (ideally 3, minimum 2)
 */
export function hasEnsembleCapability(config: ExtractionConfig): boolean {
  return getAvailableModels(config).length >= 2;
}

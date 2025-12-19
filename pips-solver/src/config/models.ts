/**
 * Model Configuration
 * Multi-model configuration for maximum accuracy extraction
 *
 * Based on December 2025 benchmarks:
 * - Gemini 2.5 Pro: Best mAP (13.3) for object/bounding box detection
 * - Claude Sonnet 4: Best structured JSON accuracy (85%)
 * - GPT-4o: Good for verification, struggles with spatial tasks
 */

// ════════════════════════════════════════════════════════════════════════════
// Model Provider Types
// ════════════════════════════════════════════════════════════════════════════

export type ModelProvider = 'anthropic' | 'google' | 'openai';

export interface ModelConfig {
  id: string;
  provider: ModelProvider;
  displayName: string;
  /** Relative strength for grid/spatial detection (0-1) */
  spatialScore: number;
  /** Relative strength for structured JSON output (0-1) */
  jsonScore: number;
  /** Relative strength for OCR/text reading (0-1) */
  ocrScore: number;
  /** Cost per 1M input tokens in USD */
  inputCostPer1M: number;
  /** Cost per 1M output tokens in USD */
  outputCostPer1M: number;
  /** Typical latency in seconds */
  typicalLatency: number;
}

// ════════════════════════════════════════════════════════════════════════════
// Available Models (December 2025)
// ════════════════════════════════════════════════════════════════════════════

export const MODELS: Record<string, ModelConfig> = {
  // Gemini models - Best for object detection and bounding boxes
  'gemini-2.5-pro': {
    id: 'gemini-2.5-pro-preview-06-05',
    provider: 'google',
    displayName: 'Gemini 2.5 Pro',
    spatialScore: 0.95, // Best mAP (13.3) among LLMs
    jsonScore: 0.85,
    ocrScore: 0.9,
    inputCostPer1M: 1.25,
    outputCostPer1M: 10.0,
    typicalLatency: 18.6,
  },
  'gemini-2.5-flash': {
    id: 'gemini-2.5-flash-preview-05-20',
    provider: 'google',
    displayName: 'Gemini 2.5 Flash',
    spatialScore: 0.85,
    jsonScore: 0.8,
    ocrScore: 0.85,
    inputCostPer1M: 0.3,
    outputCostPer1M: 2.5,
    typicalLatency: 3.0,
  },

  // Claude models - Best for structured output and reasoning
  'claude-sonnet-4': {
    id: 'claude-sonnet-4-20250514',
    provider: 'anthropic',
    displayName: 'Claude Sonnet 4',
    spatialScore: 0.75,
    jsonScore: 0.95, // Best structured JSON accuracy (85%+)
    ocrScore: 0.92,
    inputCostPer1M: 3.0,
    outputCostPer1M: 15.0,
    typicalLatency: 13.7,
  },
  'claude-3.5-sonnet': {
    id: 'claude-3-5-sonnet-20241022',
    provider: 'anthropic',
    displayName: 'Claude 3.5 Sonnet',
    spatialScore: 0.7,
    jsonScore: 0.9,
    ocrScore: 0.9,
    inputCostPer1M: 3.0,
    outputCostPer1M: 15.0,
    typicalLatency: 10.0,
  },

  // OpenAI models - Good general purpose, weaker spatial
  'gpt-4o': {
    id: 'gpt-4o-2024-11-20',
    provider: 'openai',
    displayName: 'GPT-4o',
    spatialScore: 0.58, // Only 58% on geometric tasks per research
    jsonScore: 0.78,
    ocrScore: 0.88,
    inputCostPer1M: 2.5,
    outputCostPer1M: 10.0,
    typicalLatency: 6.0,
  },
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Task-Optimized Model Selection
// ════════════════════════════════════════════════════════════════════════════

export type ExtractionTask =
  | 'grid_detection' // Finding grid lines and cell boundaries
  | 'region_colors' // Identifying colored regions
  | 'pip_counting' // Counting dots on dominoes
  | 'constraint_ocr' // Reading constraint text (=12, <10, etc)
  | 'verification' // Cross-checking extraction results
  | 'domino_detection'; // Finding dominoes in tray

/**
 * Optimal model for each task based on benchmarks
 */
export const TASK_OPTIMAL_MODELS: Record<ExtractionTask, keyof typeof MODELS> = {
  grid_detection: 'gemini-2.5-pro', // Best spatial/bounding box
  region_colors: 'gemini-2.5-pro', // Good at color segmentation
  pip_counting: 'gemini-2.5-pro', // Best object detection
  constraint_ocr: 'claude-sonnet-4', // Best text interpretation
  verification: 'claude-sonnet-4', // Best reasoning
  domino_detection: 'gemini-2.5-pro', // Best object detection
};

/**
 * Fallback chain for each task if primary fails
 */
export const TASK_FALLBACK_CHAIN: Record<ExtractionTask, (keyof typeof MODELS)[]> = {
  grid_detection: ['gemini-2.5-pro', 'gemini-2.5-flash', 'claude-sonnet-4'],
  region_colors: ['gemini-2.5-pro', 'claude-sonnet-4', 'gemini-2.5-flash'],
  pip_counting: ['gemini-2.5-pro', 'gemini-2.5-flash', 'claude-sonnet-4'],
  constraint_ocr: ['claude-sonnet-4', 'gemini-2.5-pro', 'claude-3.5-sonnet'],
  verification: ['claude-sonnet-4', 'gemini-2.5-pro', 'claude-3.5-sonnet'],
  domino_detection: ['gemini-2.5-pro', 'gemini-2.5-flash', 'claude-sonnet-4'],
};

// ════════════════════════════════════════════════════════════════════════════
// Extraction Strategies
// ════════════════════════════════════════════════════════════════════════════

export type ExtractionStrategy =
  | 'fast' // Single model, fastest response
  | 'balanced' // Task-optimal models, good accuracy
  | 'accurate' // Multi-model with verification
  | 'ensemble'; // Query multiple models, use consensus

export interface StrategyConfig {
  /** Primary models to use for extraction */
  primaryModels: (keyof typeof MODELS)[];
  /** Whether to run verification pass */
  enableVerification: boolean;
  /** Confidence threshold to trigger re-extraction */
  confidenceThreshold: number;
  /** Whether to use ensemble voting */
  useEnsemble: boolean;
  /** Number of models to query for ensemble */
  ensembleSize: number;
  /** Maximum retries per model */
  maxRetries: number;
}

export const STRATEGIES: Record<ExtractionStrategy, StrategyConfig> = {
  fast: {
    primaryModels: ['gemini-2.5-flash'],
    enableVerification: false,
    confidenceThreshold: 0.7,
    useEnsemble: false,
    ensembleSize: 1,
    maxRetries: 1,
  },
  balanced: {
    primaryModels: ['gemini-2.5-pro'],
    enableVerification: true,
    confidenceThreshold: 0.85,
    useEnsemble: false,
    ensembleSize: 1,
    maxRetries: 2,
  },
  accurate: {
    primaryModels: ['gemini-2.5-pro', 'claude-sonnet-4'],
    enableVerification: true,
    confidenceThreshold: 0.9,
    useEnsemble: false,
    ensembleSize: 1,
    maxRetries: 3,
  },
  ensemble: {
    primaryModels: ['gemini-2.5-pro', 'claude-sonnet-4', 'gemini-2.5-flash'],
    enableVerification: true,
    confidenceThreshold: 0.95,
    useEnsemble: true,
    ensembleSize: 3,
    maxRetries: 2,
  },
};

// ════════════════════════════════════════════════════════════════════════════
// Legacy Support (for existing code)
// ════════════════════════════════════════════════════════════════════════════

/** @deprecated Use MODELS and TASK_OPTIMAL_MODELS instead */
export const MODEL_CANDIDATES = [
  'claude-sonnet-4-20250514',
  'claude-3-5-sonnet-20241022',
  'claude-3-opus-20240229',
] as const;

export type ModelId = (typeof MODEL_CANDIDATES)[number];
export const DEFAULT_MODEL = MODEL_CANDIDATES[0];

// ════════════════════════════════════════════════════════════════════════════
// API Key Configuration
// ════════════════════════════════════════════════════════════════════════════

export interface APIKeys {
  anthropic?: string;
  google?: string;
  openai?: string;
}

/**
 * Check which providers have valid API keys configured
 */
export function getAvailableProviders(keys: APIKeys): ModelProvider[] {
  const available: ModelProvider[] = [];
  if (keys.anthropic?.trim()) available.push('anthropic');
  if (keys.google?.trim()) available.push('google');
  if (keys.openai?.trim()) available.push('openai');
  return available;
}

/**
 * Get best available model for a task given available API keys
 */
export function getBestModelForTask(
  task: ExtractionTask,
  availableProviders: ModelProvider[]
): ModelConfig | null {
  const fallbackChain = TASK_FALLBACK_CHAIN[task];

  for (const modelKey of fallbackChain) {
    const model = MODELS[modelKey];
    if (availableProviders.includes(model.provider)) {
      return model;
    }
  }

  return null;
}

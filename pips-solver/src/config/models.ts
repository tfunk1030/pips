/**
 * Model Configuration
 * Multi-model configuration for maximum accuracy extraction
 *
 * Updated December 21, 2025:
 * - Gemini 3 Pro Preview: Best for grid/spatial detection
 * - GPT-5.2: Best for OCR, pip counting, and fine visual detail
 * - Claude Opus 4.5: Best for instruction following and structured JSON output
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
// Available Models (Updated December 21, 2025)
// ════════════════════════════════════════════════════════════════════════════

export const MODELS: Record<string, ModelConfig> = {
  // Gemini 3 Pro Preview - Latest Google flagship (Dec 2025)
  'gemini-3-pro-preview': {
    id: 'google/gemini-3-pro-preview',
    provider: 'google',
    displayName: 'Gemini 3 Pro',
    spatialScore: 0.97, // Best for grid geometry and spatial understanding
    jsonScore: 0.90,
    ocrScore: 0.94,
    inputCostPer1M: 1.25,
    outputCostPer1M: 5.0,
    typicalLatency: 12.0,
  },

  // GPT-5.2 - Latest OpenAI flagship (Dec 2025)
  'gpt-5.2': {
    id: 'openai/gpt-5.2',
    provider: 'openai',
    displayName: 'GPT-5.2',
    spatialScore: 0.88,
    jsonScore: 0.92,
    ocrScore: 0.96, // Best for OCR and pip counting
    inputCostPer1M: 1.75,
    outputCostPer1M: 14.0,
    typicalLatency: 8.0,
  },

  // Claude Opus 4.5 - Latest Anthropic flagship (Dec 2025)
  'claude-opus-4.5': {
    id: 'anthropic/claude-opus-4.5',
    provider: 'anthropic',
    displayName: 'Claude Opus 4.5',
    spatialScore: 0.85,
    jsonScore: 0.98, // Best for structured JSON and validation
    ocrScore: 0.94,
    inputCostPer1M: 5.0,
    outputCostPer1M: 25.0,
    typicalLatency: 10.0,
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
 * Optimal model for each task based on benchmarks (December 2025)
 *
 * Models:
 * - gemini-3-pro-preview: Best for grid/spatial detection (spatialScore: 0.97)
 * - gpt-5.2: Best for OCR and pip counting (ocrScore: 0.96)
 * - claude-opus-4.5: Best for structured JSON output (jsonScore: 0.98)
 */
export const TASK_OPTIMAL_MODELS: Record<ExtractionTask, keyof typeof MODELS> = {
  grid_detection: 'gemini-3-pro-preview', // Best spatial/grid detection
  region_colors: 'gemini-3-pro-preview', // Best image understanding
  pip_counting: 'gpt-5.2', // Best OCR for counting dots
  constraint_ocr: 'gpt-5.2', // Best text reading
  verification: 'claude-opus-4.5', // Best reasoning and JSON
  domino_detection: 'gpt-5.2', // Best object detection/OCR
};

/**
 * Fallback chain for each task if primary fails
 */
export const TASK_FALLBACK_CHAIN: Record<ExtractionTask, (keyof typeof MODELS)[]> = {
  grid_detection: ['gemini-3-pro-preview', 'gpt-5.2', 'claude-opus-4.5'],
  region_colors: ['gemini-3-pro-preview', 'gpt-5.2', 'claude-opus-4.5'],
  pip_counting: ['gpt-5.2', 'gemini-3-pro-preview', 'claude-opus-4.5'],
  constraint_ocr: ['gpt-5.2', 'claude-opus-4.5', 'gemini-3-pro-preview'],
  verification: ['claude-opus-4.5', 'gpt-5.2', 'gemini-3-pro-preview'],
  domino_detection: ['gpt-5.2', 'gemini-3-pro-preview', 'claude-opus-4.5'],
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
    primaryModels: ['gemini-3-pro-preview'],
    enableVerification: false,
    confidenceThreshold: 0.6,
    useEnsemble: false,
    ensembleSize: 1,
    maxRetries: 1,
  },
  balanced: {
    primaryModels: ['gpt-5.2'],
    enableVerification: true,
    confidenceThreshold: 0.75,
    useEnsemble: false,
    ensembleSize: 1,
    maxRetries: 1,
  },
  accurate: {
    primaryModels: ['gpt-5.2', 'claude-opus-4.5'],
    enableVerification: true,
    confidenceThreshold: 0.85,
    useEnsemble: false,
    ensembleSize: 1,
    maxRetries: 1,
  },
  ensemble: {
    primaryModels: ['gemini-3-pro-preview', 'gpt-5.2', 'claude-opus-4.5'],
    enableVerification: true,
    confidenceThreshold: 0.90,
    useEnsemble: true,
    ensembleSize: 3,
    maxRetries: 1, // Reduced to 1 for faster overall extraction
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

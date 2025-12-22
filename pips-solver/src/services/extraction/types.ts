/**
 * Multi-Stage Extraction Pipeline Types
 *
 * Defines all interfaces for the 5-stage extraction pipeline with
 * 3-model ensemble (Gemini 3 Pro, GPT-5.2, Claude Opus 4.5).
 */

// =============================================================================
// API Configuration
// =============================================================================

export interface ApiKeys {
  /** OpenRouter API key - if set, used for all models */
  openrouter?: string;
  /** Google API key - override for Gemini */
  google?: string;
  /** OpenAI API key - override for GPT */
  openai?: string;
  /** Anthropic API key - override for Claude */
  anthropic?: string;
}

export interface ExtractionConfig {
  /** API keys - OpenRouter OR individual provider keys */
  apiKeys: ApiKeys;

  /** Model identifiers (OpenRouter format) */
  models: {
    /** Model for all stages - default: "google/gemini-3-pro" */
    gemini: string;
    /** Model for all stages - default: "openai/gpt-5.2" */
    gpt: string;
    /** Model for all stages - default: "anthropic/claude-opus-4.5" */
    claude: string;
  };

  /** Maximum retries per stage when consensus fails */
  maxRetries: number;

  /** Confidence difference threshold for majority fallback (default: 0.10) */
  confidenceThreshold: number;

  /** Timeout per API call in milliseconds */
  timeoutMs: number;

  /** NYT-specific validation bounds */
  validation: {
    minGridSize: number;
    maxGridSize: number;
    pipRange: [number, number];
    uniqueDominoes: boolean;
  };

  /** Flag results with confidence below this threshold */
  lowConfidenceThreshold: number;

  /** Save raw model responses for debugging */
  saveDebugResponses: boolean;

  /** Fall back to legacy extraction on failure */
  enableLegacyFallback: boolean;
}

// =============================================================================
// Model Response Types
// =============================================================================

export interface ModelResponse<T = unknown> {
  /** Model identifier (e.g., "google/gemini-3-pro") */
  model: string;
  /** Parsed answer from the model */
  answer: T;
  /** Model's self-reported confidence (0.0-1.0) */
  confidence: number;
  /** API call latency in milliseconds */
  latencyMs: number;
  /** Raw response text (for debugging) */
  rawResponse?: string;
  /** Error if the call failed */
  error?: string;
}

export interface ConsensusResult<T = unknown> {
  /** The consensus answer */
  answer: T;
  /** How consensus was reached */
  source: 'confidence' | 'majority' | 'best-effort';
  /** Whether consensus was confident (not best-effort) */
  confident: boolean;
  /** All model responses used */
  responses: ModelResponse<T>[];
  /** Number of retries attempted */
  retryCount: number;
}

// =============================================================================
// Stage-Specific Types
// =============================================================================

/** Stage 1: Grid Geometry */
export interface GridGeometryResult {
  rows: number;
  cols: number;
  confidence: number;
}

/** Stage 2: Cell/Hole Detection */
export interface CellDetectionResult {
  /** Multiline string of '.' (cell) and '#' (hole) */
  shape: string;
  confidence: number;
}

/** Stage 3: Region Mapping */
export interface RegionMappingResult {
  /** Multiline string of region labels (A-Z) and '#' for holes */
  regions: string;
  confidence: number;
}

/** Stage 4: Constraint Extraction */
export interface Constraint {
  type: 'sum' | 'all_equal';
  op?: '==' | '<' | '>';
  value?: number;
}

export interface ConstraintExtractionResult {
  constraints: Record<string, Constraint>;
  confidence: number;
}

/** Stage 5: Domino Extraction */
export interface DominoExtractionResult {
  /** Array of domino tiles as [pip1, pip2] pairs */
  dominoes: [number, number][];
  confidence: number;
}

// =============================================================================
// Pipeline Types
// =============================================================================

export type StageName = 'grid' | 'cells' | 'regions' | 'constraints' | 'dominoes';

export interface StageConfidence {
  grid: number;
  cells: number;
  regions: number;
  constraints: number;
  dominoes: number;
}

export interface ExtractionResult {
  /** Core puzzle data */
  grid: {
    rows: number;
    cols: number;
    /** Multiline string of '.' and '#' */
    shape: string;
    /** Multiline string of region labels and '#' */
    regions: string;
  };

  /** Region constraints */
  constraints: Record<string, Constraint>;

  /** Domino tiles as [pip1, pip2] pairs */
  dominoes: [number, number][];

  /** Confidence scores */
  confidence: {
    /** Minimum of all stage confidences */
    overall: number;
    /** Per-stage confidence breakdown */
    perStage: StageConfidence;
  };

  /** Whether manual review is recommended */
  needsReview: boolean;

  /** Hints about what to review */
  reviewHints: string[];

  /** Debug information (if saveDebugResponses enabled) */
  debug?: {
    rawResponses: Record<StageName, ModelResponse[]>;
    consensusDetails: Record<StageName, ConsensusResult>;
    totalRetries: number;
    totalLatencyMs: number;
  };
}

// =============================================================================
// Validation Types
// =============================================================================

export interface ValidationError {
  stage: StageName;
  field: string;
  message: string;
  value?: unknown;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}

// =============================================================================
// API Client Types
// =============================================================================

export type ApiProvider = 'openrouter' | 'google' | 'openai' | 'anthropic';

export interface ApiEndpoint {
  provider: ApiProvider;
  endpoint: string;
  key: string;
  model: string;
}

export interface VisionApiRequest {
  model: string;
  imageBase64: string;
  prompt: string;
  maxTokens?: number;
}

export interface VisionApiResponse {
  content: string;
  latencyMs: number;
  error?: string;
}

/**
 * Hint Service
 *
 * API client for the graduated hint system.
 * Provides functions to request hints from the pips-agent backend.
 */

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

/** Hint levels from 1 (general) to 4 (specific) */
export type HintLevel = 1 | 2 | 3 | 4;

/** Types of hints returned by the API */
export type HintType = 'strategy' | 'direction' | 'cell' | 'partial_solution';

/** Individual cell placement info for Level 3/4 hints */
export interface CellPlacement {
  row: number;
  col: number;
  value: number;
  region?: string;
}

/** Hint content returned from the API */
export interface HintContent {
  level: HintLevel;
  type: HintType;
  content: string;
  region?: string;
  cell?: { row: number; col: number };
  cells?: CellPlacement[];
}

/** Pips configuration for puzzle spec */
export interface PipsConfig {
  pip_min: number;
  pip_max: number;
}

/** Dominoes configuration for puzzle spec */
export interface DominoesConfig {
  tiles: number[][];
  unique?: boolean;
}

/** Board configuration for puzzle spec */
export interface BoardConfig {
  shape: string[];
  regions: string[];
}

/** Region constraint definition */
export interface RegionConstraint {
  type: 'sum' | 'all_equal';
  op?: '==' | '!=' | '<' | '>' | '<=' | '>=';
  value?: number;
}

/** Complete puzzle specification */
export interface PuzzleSpec {
  pips: PipsConfig;
  dominoes: DominoesConfig;
  board: BoardConfig;
  region_constraints: Record<string, RegionConstraint>;
}

/** Request payload for hint generation */
export interface HintRequest {
  puzzle_spec: PuzzleSpec;
  level: HintLevel;
  current_state?: Record<string, unknown>;
}

/** Response from the hint generation API */
export interface HintResponse {
  success: boolean;
  hint?: HintContent;
  error?: string;
}

/** Error thrown when hint generation fails */
export class HintServiceError extends Error {
  constructor(
    message: string,
    public readonly code?: string,
    public readonly isNetworkError?: boolean
  ) {
    super(message);
    this.name = 'HintServiceError';
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

/** Default API configuration */
const DEFAULT_CONFIG = {
  /** Base URL for the pips-agent API */
  baseUrl: 'http://localhost:8001',
  /** Request timeout in milliseconds */
  timeout: 30000,
} as const;

/** Current API configuration */
let apiConfig = { ...DEFAULT_CONFIG };

/**
 * Configure the hint service with custom settings.
 *
 * @param config - Partial configuration to merge with defaults
 */
export function configureHintService(config: Partial<typeof DEFAULT_CONFIG>): void {
  apiConfig = { ...apiConfig, ...config };
}

/**
 * Get the current API configuration.
 */
export function getHintServiceConfig(): typeof DEFAULT_CONFIG {
  return { ...apiConfig };
}

// ════════════════════════════════════════════════════════════════════════════
// API Client
// ════════════════════════════════════════════════════════════════════════════

/**
 * Generate a hint for the given puzzle at the specified level.
 *
 * @param puzzleSpec - The current puzzle specification
 * @param level - The hint level to request (1-4)
 * @param currentState - Optional current puzzle state with user placements
 * @returns Promise resolving to the hint content
 * @throws HintServiceError if the request fails
 *
 * @example
 * ```typescript
 * const hint = await generateHint(puzzleSpec, 1);
 * console.log(hint.content); // "Consider looking for regions with..."
 * ```
 */
export async function generateHint(
  puzzleSpec: PuzzleSpec,
  level: HintLevel,
  currentState?: Record<string, unknown>
): Promise<HintContent> {
  const url = `${apiConfig.baseUrl}/generate-hint`;

  const requestBody: HintRequest = {
    puzzle_spec: puzzleSpec,
    level,
    ...(currentState && { current_state: currentState }),
  };

  // Create abort controller for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), apiConfig.timeout);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    // Handle HTTP errors
    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      throw new HintServiceError(
        `Server error: ${response.status} ${response.statusText}`,
        `HTTP_${response.status}`,
        false
      );
    }

    // Parse response
    const data: HintResponse = await response.json();

    // Handle API-level errors
    if (!data.success) {
      throw new HintServiceError(
        data.error ?? 'Hint generation failed',
        'API_ERROR',
        false
      );
    }

    // Validate hint content
    if (!data.hint) {
      throw new HintServiceError(
        'Server returned success but no hint content',
        'EMPTY_RESPONSE',
        false
      );
    }

    return data.hint;
  } catch (error) {
    clearTimeout(timeoutId);

    // Re-throw HintServiceError as-is
    if (error instanceof HintServiceError) {
      throw error;
    }

    // Handle abort (timeout)
    if (error instanceof Error && error.name === 'AbortError') {
      throw new HintServiceError(
        'Request timed out. Please try again.',
        'TIMEOUT',
        true
      );
    }

    // Handle network errors
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new HintServiceError(
        'Unable to connect to hint server. Please check your connection.',
        'NETWORK_ERROR',
        true
      );
    }

    // Handle other errors
    throw new HintServiceError(
      error instanceof Error ? error.message : 'An unexpected error occurred',
      'UNKNOWN',
      false
    );
  }
}

/**
 * Check if the hint service is available.
 *
 * @returns Promise resolving to true if the service is healthy
 */
export async function checkHintServiceHealth(): Promise<boolean> {
  const url = `${apiConfig.baseUrl}/health`;

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });

    if (!response.ok) {
      return false;
    }

    const data = await response.json();
    return data.status === 'healthy';
  } catch {
    return false;
  }
}

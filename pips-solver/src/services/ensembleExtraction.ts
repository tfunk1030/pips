/**
 * Ensemble Extraction Service
 * Maximum accuracy extraction using multi-model consensus and cross-validation
 *
 * Strategy:
 * 1. Query multiple models in parallel
 * 2. Cross-validate results using voting/consensus
 * 3. Use verification pass for low-confidence results
 * 4. Return highest-confidence result with provenance
 */

import { z } from 'zod';
import { APIKeys, ExtractionStrategy, MODELS, STRATEGIES } from '../config/models';
import {
  AIExtractionResult,
  BoardExtractionResult,
  BoardModelResponse,
  DominoExtractionResult,
  DominoModelResponse,
  DominoPair,
  RawResponses,
} from '../model/overlayTypes';
import {
  callMultipleModels,
  callVisionModel,
  inferMediaType,
  normalizeBase64,
} from './modelClients';

// ════════════════════════════════════════════════════════════════════════════
// Optimized Prompts (based on December 2025 research)
// ════════════════════════════════════════════════════════════════════════════

/**
 * System prompt for Gemini (best at spatial/bounding box tasks)
 */
const GEMINI_SYSTEM_PROMPT = `You are an expert at analyzing game screenshots with pixel-perfect accuracy.
Your responses MUST be valid JSON only - no markdown, no code blocks, no explanations.
For bounding boxes, use normalized coordinates (0-1000 scale).
Be precise with counting and spatial relationships.`;

/**
 * System prompt for Claude (best at structured output and reasoning)
 */
const CLAUDE_SYSTEM_PROMPT = `You are an expert puzzle analyzer. Your task is to extract structured data from game screenshots.
Always respond with valid JSON only. Never use markdown formatting.
Think step-by-step but only output the final JSON result.
Be conservative with confidence scores - lower is better than overconfident.`;

/**
 * Optimized board extraction prompt (Gemini-tuned)
 */
const BOARD_EXTRACTION_PROMPT_V2 = `Analyze this Pips puzzle image. This is a CROPPED image showing just the puzzle grid.

CRITICAL STEPS:

1. GRID SIZE: Count rows and columns carefully. Common sizes: 4x4, 5x5, 6x6.

2. HOLES: Check ALL 4 CORNERS for missing cells. Mark holes as "#" in both shape AND regions.
   Example 4x4 with corner holes: "...#\\n....\\n....\\n...#"

3. REGIONS - IMPORTANT:
   - Look at the BACKGROUND COLOR of each cell (not the dotted border color)
   - Cells with the SAME background color = same region = same letter
   - Assign letters A, B, C, D... to distinct colors
   - Be CONSISTENT: if two cells look the same color, they MUST have the same letter
   - Common colors: orange, green, blue, purple, pink, yellow
   - Put "#" for holes in both shape and regions strings

4. CONSTRAINTS: Look for numbers/symbols near or on regions:
   - Number like "12" or "=12" → type: "sum", op: "==", value: 12
   - "<10" → type: "sum", op: "<", value: 10
   - "=" symbol only → type: "all_equal"
   - "≠" or different symbol → type: "all_different"

OUTPUT JSON only (no explanation outside JSON):
{
  "rows": 4,
  "cols": 4,
  "shape": "...#\\n....\\n....\\n...#",
  "regions": "AAB#\\nAABC\\nDDBC\\nDD.#",
  "constraints": {"A": {"type": "sum", "op": "==", "value": 8}, "B": {"type": "all_different"}},
  "confidence": {"grid": 0.9, "regions": 0.85, "constraints": 0.9},
  "reasoning": "4x4, holes at (0,3) and (3,3), 4 regions by color"
}`;

/**
 * Optimized domino extraction prompt
 */
const DOMINO_EXTRACTION_PROMPT_V2 = `This image shows the DOMINO TRAY from a Pips puzzle.

Find all dominoes in this image. Each domino has TWO halves with 0-6 pips (dots) each.

PIP COUNTING (be precise!):
- 0: Blank/empty half
- 1: Single dot in center
- 2: Two dots diagonally
- 3: Three dots in diagonal line
- 4: Four dots in corners
- 5: Four corners + one center
- 6: Six dots (2 columns of 3)

PROCESS:
1. Locate each domino (rectangular with dividing line)
2. Count pips on LEFT/TOP half
3. Count pips on RIGHT/BOTTOM half
4. Record as [first, second]

Dominoes may be horizontal or vertical. There are typically 7-8 dominoes.

OUTPUT JSON only:
{
  "dominoes": [[0, 1], [2, 3], [4, 5], [6, 6], [1, 2], [3, 4], [5, 6]],
  "confidence": 0.95,
  "reasoning": "7 dominoes found, all pips clearly visible"
}`;

/**
 * Verification prompt for cross-checking results
 */
const VERIFICATION_PROMPT_V2 = `Verify this puzzle extraction against the original image.

PREVIOUS EXTRACTION:
- Grid: {rows}x{cols}
- Shape:
{shape}

- Regions:
{regions}

- Constraints: {constraints}

- Dominoes: {dominoes}

VERIFICATION CHECKLIST:
1. Are grid dimensions correct? Count rows and columns.
2. Are holes (#) in the correct positions?
3. Are region colors mapped correctly?
4. Are constraint values accurate?
5. Are all dominoes captured with correct pip counts?

OUTPUT (JSON only):
{
  "verified": true,
  "issues": [],
  "corrections": null
}

OR if corrections needed:
{
  "verified": false,
  "issues": ["Region B should extend to row 3", "Domino [4,2] should be [4,3]"],
  "corrections": {
    "regions": "corrected region string if needed",
    "dominoes": [[4, 3], ...],
    "constraints": { ... }
  }
}`;

// ════════════════════════════════════════════════════════════════════════════
// Zod Schemas
// ════════════════════════════════════════════════════════════════════════════

// Lenient constraint schema - normalizes variations in model output
const ConstraintSchema = z.object({
  // Accept various type formats that models might return
  type: z.string().transform(t => {
    const normalized = t.toLowerCase().replace(/[_\s]/g, '');
    if (normalized === 'sum' || normalized === 'total') return 'sum';
    if (normalized.includes('equal') && !normalized.includes('diff')) return 'all_equal';
    if (normalized.includes('diff') || normalized.includes('unique')) return 'all_different';
    return t; // Keep original if unknown - will be handled downstream
  }),
  op: z
    .enum(['==', '<', '>', '!='])
    .optional()
    .nullable()
    .transform(v => v ?? undefined),
  value: z
    .number()
    .optional()
    .nullable()
    .transform(v => v ?? undefined),
});

const BoardExtractionSchema = z.object({
  rows: z.number().min(2).max(12),
  cols: z.number().min(2).max(12),
  gridLocation: z
    .object({
      left: z.number(),
      top: z.number(),
      right: z.number(),
      bottom: z.number(),
      imageWidth: z.number(),
      imageHeight: z.number(),
    })
    .optional(),
  shape: z.string().min(1),
  regions: z.string().min(1),
  constraints: z.record(z.string(), ConstraintSchema).optional(),
  confidence: z
    .object({
      grid: z.number().min(0).max(1),
      regions: z.number().min(0).max(1),
      constraints: z.number().min(0).max(1),
    })
    .optional(),
  reasoning: z.string().optional(),
});

const DominoExtractionSchema = z.object({
  dominoes: z.array(z.tuple([z.number().min(0).max(6), z.number().min(0).max(6)])),
  confidence: z.number().min(0).max(1).optional(),
  reasoning: z.string().optional(),
});

const VerificationSchema = z.object({
  verified: z.boolean(),
  issues: z.array(z.string()),
  corrections: z
    .object({
      rows: z.number().optional(),
      cols: z.number().optional(),
      shape: z.string().optional(),
      regions: z.string().optional(),
      constraints: z.record(z.string(), ConstraintSchema).optional(),
      dominoes: z.array(z.tuple([z.number(), z.number()])).optional(),
    })
    .nullable()
    .optional(),
});

// ════════════════════════════════════════════════════════════════════════════
// Progress Tracking
// ════════════════════════════════════════════════════════════════════════════

export interface ExtractionProgress {
  step:
    | 'initializing'
    | 'board_primary'
    | 'board_secondary'
    | 'board_verification'
    | 'dominoes_primary'
    | 'dominoes_secondary'
    | 'cross_validation'
    | 'complete';
  message: string;
  modelsUsed?: string[];
  confidence?: number;
}

// ════════════════════════════════════════════════════════════════════════════
// JSON Parsing Utilities
// ════════════════════════════════════════════════════════════════════════════

function extractJSON(text: string): string | null {
  // Try to find JSON object in response
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return null;

  let jsonStr = match[0];

  // Fix common issues: multi-line strings in shape/regions
  const fixField = (fieldName: string, json: string): string => {
    const pattern = new RegExp(`"${fieldName}"\\s*:\\s*"([^"]*)"\\s*\\n\\s*"([^"]*)"`, 'g');
    let fixed = json;
    let prev = '';
    let iter = 0;
    while (fixed !== prev && iter < 10) {
      prev = fixed;
      fixed = fixed.replace(pattern, (_, l1, l2) => `"${fieldName}": "${l1}\\n${l2}"`);
      iter++;
    }
    return fixed;
  };

  jsonStr = fixField('shape', jsonStr);
  jsonStr = fixField('regions', jsonStr);

  return jsonStr;
}

function parseJSONSafely<T>(
  text: string,
  schema: z.ZodSchema<T>
): { success: true; data: T } | { success: false; error: string } {
  const jsonStr = extractJSON(text);
  if (!jsonStr) {
    return { success: false, error: 'No JSON found in response' };
  }

  try {
    const parsed = JSON.parse(jsonStr);
    const validated = schema.safeParse(parsed);
    if (!validated.success) {
      const errors = validated.error.issues.map(e => `${e.path.join('.')}: ${e.message}`);
      return { success: false, error: `Validation failed: ${errors.join(', ')}` };
    }
    return { success: true, data: validated.data };
  } catch (e) {
    return { success: false, error: `JSON parse error: ${e}` };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Consensus Logic
// ════════════════════════════════════════════════════════════════════════════

interface BoardResult {
  data: BoardExtractionResult;
  confidence: number;
  model: string;
}

interface DominoResult {
  data: DominoExtractionResult;
  confidence: number;
  model: string;
}

/**
 * Compare two board extractions for consistency
 */
function compareBoardResults(a: BoardExtractionResult, b: BoardExtractionResult): number {
  let score = 0;
  const total = 4;

  // Grid dimensions
  if (a.rows === b.rows && a.cols === b.cols) score += 1;

  // Shape (normalized comparison)
  const normalizeGrid = (s: string) => s.replace(/\\n/g, '\n').trim();
  if (normalizeGrid(a.shape) === normalizeGrid(b.shape)) score += 1;

  // Regions
  if (normalizeGrid(a.regions) === normalizeGrid(b.regions)) score += 1;

  // Constraints (key match)
  const aKeys = Object.keys(a.constraints || {})
    .sort()
    .join(',');
  const bKeys = Object.keys(b.constraints || {})
    .sort()
    .join(',');
  if (aKeys === bKeys) score += 1;

  return score / total;
}

/**
 * Compare two domino extractions for consistency
 */
function compareDominoResults(a: DominoExtractionResult, b: DominoExtractionResult): number {
  const normalize = (d: DominoPair): string => {
    const sorted = [...d].sort((x, y) => x - y);
    return `${sorted[0]}-${sorted[1]}`;
  };

  const aSet = new Set(a.dominoes.map(normalize));
  const bSet = new Set(b.dominoes.map(normalize));

  if (aSet.size === 0 && bSet.size === 0) return 1;
  if (aSet.size === 0 || bSet.size === 0) return 0;

  let matches = 0;
  for (const d of aSet) {
    if (bSet.has(d)) matches++;
  }

  const union = new Set([...aSet, ...bSet]).size;
  return matches / union;
}

/**
 * Select best result from ensemble based on confidence and consensus
 */
function selectBestBoard(results: BoardResult[]): BoardResult {
  if (results.length === 1) return results[0];

  // Score each result by confidence + agreement with others
  const scored = results.map((r, i) => {
    let consensusScore = 0;
    for (let j = 0; j < results.length; j++) {
      if (i !== j) {
        consensusScore += compareBoardResults(r.data, results[j].data);
      }
    }
    consensusScore /= results.length - 1;

    // Combined score: 60% confidence, 40% consensus
    const combinedScore = r.confidence * 0.6 + consensusScore * 0.4;
    return { result: r, combinedScore, consensusScore };
  });

  scored.sort((a, b) => b.combinedScore - a.combinedScore);

  console.log(
    '[Ensemble] Board results scored:',
    scored.map(s => ({
      model: s.result.model,
      confidence: s.result.confidence.toFixed(2),
      consensus: s.consensusScore.toFixed(2),
      combined: s.combinedScore.toFixed(2),
    }))
  );

  return scored[0].result;
}

function selectBestDominoes(results: DominoResult[]): DominoResult {
  if (results.length === 1) return results[0];

  const scored = results.map((r, i) => {
    let consensusScore = 0;
    for (let j = 0; j < results.length; j++) {
      if (i !== j) {
        consensusScore += compareDominoResults(r.data, results[j].data);
      }
    }
    consensusScore /= results.length - 1;

    const combinedScore = r.confidence * 0.6 + consensusScore * 0.4;
    return { result: r, combinedScore, consensusScore };
  });

  scored.sort((a, b) => b.combinedScore - a.combinedScore);

  console.log(
    '[Ensemble] Domino results scored:',
    scored.map(s => ({
      model: s.result.model,
      confidence: s.result.confidence.toFixed(2),
      consensus: s.consensusScore.toFixed(2),
      combined: s.combinedScore.toFixed(2),
    }))
  );

  return scored[0].result;
}

// ════════════════════════════════════════════════════════════════════════════
// Main Extraction Functions
// ════════════════════════════════════════════════════════════════════════════

export interface EnsembleExtractionOptions {
  strategy: ExtractionStrategy;
  onProgress?: (progress: ExtractionProgress) => void;
  /** When true, stores raw per-model responses for comparison/debugging */
  saveDebugResponses?: boolean;
}

/**
 * Extract board structure with ensemble approach
 */
async function extractBoardEnsemble(
  base64Image: string,
  keys: APIKeys,
  strategy: ExtractionStrategy,
  onProgress?: (progress: ExtractionProgress) => void,
  saveDebugResponses?: boolean
): Promise<{
  success: boolean;
  data?: BoardExtractionResult;
  error?: string;
  modelsUsed: string[];
  rawResponses?: BoardModelResponse[];
  selectedModel?: string;
}> {
  const config = STRATEGIES[strategy];
  const modelsUsed: string[] = [];
  const results: BoardResult[] = [];
  const rawResponses: BoardModelResponse[] = [];

  const normalized = normalizeBase64(base64Image);
  const mediaType = inferMediaType(base64Image);

  // Determine which models to use
  const modelsToQuery = config.useEnsemble
    ? config.primaryModels.slice(0, config.ensembleSize)
    : [config.primaryModels[0]];

  onProgress?.({
    step: 'board_primary',
    message: `Extracting board with ${modelsToQuery.length} model(s)...`,
    modelsUsed: modelsToQuery,
  });

  // Query models in parallel
  const queryStartTime = Date.now();
  const responses = await callMultipleModels(
    keys,
    modelsToQuery as (keyof typeof MODELS)[],
    [
      {
        role: 'system',
        content: [{ type: 'text', text: GEMINI_SYSTEM_PROMPT }],
      },
      {
        role: 'user',
        content: [
          { type: 'image', base64: normalized, mediaType },
          { type: 'text', text: BOARD_EXTRACTION_PROMPT_V2 },
        ],
      },
    ],
    { temperature: 0.1, jsonMode: true, maxTokens: 16384 }
  );
  const queryEndTime = Date.now();
  const responseMs = queryEndTime - queryStartTime;

  // Process results
  for (const resp of responses) {
    modelsUsed.push(resp.modelKey);

    if (resp.result) {
      const parseStartTime = Date.now();
      const parsed = parseJSONSafely(resp.result.text, BoardExtractionSchema);
      const parseMs = Date.now() - parseStartTime;

      if (parsed.success) {
        const confidence = parsed.data.confidence
          ? (parsed.data.confidence.grid +
              parsed.data.confidence.regions +
              parsed.data.confidence.constraints) /
            3
          : 0.8;

        const boardData: BoardExtractionResult = {
          rows: parsed.data.rows,
          cols: parsed.data.cols,
          shape: parsed.data.shape,
          regions: parsed.data.regions,
          constraints: parsed.data.constraints || {},
          confidence: parsed.data.confidence || { grid: 0.8, regions: 0.8, constraints: 0.8 },
          gridLocation: parsed.data.gridLocation,
        };

        results.push({
          data: boardData,
          confidence,
          model: resp.modelKey,
        });

        // Store raw response for debug/comparison
        if (saveDebugResponses) {
          rawResponses.push({
            model: resp.modelKey,
            rawText: resp.result.text,
            parsedData: boardData,
            parseSuccess: true,
            timing: {
              responseMs: responseMs / responses.length, // Approximate per-model time
              parseMs,
            },
            confidence,
          });
        }
      } else {
        // Store failed parse for debug
        if (saveDebugResponses) {
          rawResponses.push({
            model: resp.modelKey,
            rawText: resp.result.text,
            parsedData: null,
            parseSuccess: false,
            parseError: parsed.error,
            timing: {
              responseMs: responseMs / responses.length,
              parseMs: Date.now() - parseStartTime,
            },
          });
        }
      }
    } else {
      // Store failed call for debug
      if (saveDebugResponses) {
        rawResponses.push({
          model: resp.modelKey,
          rawText: '',
          parsedData: null,
          parseSuccess: false,
          parseError: resp.error || 'Model call failed',
          timing: {
            responseMs: responseMs / responses.length,
            parseMs: 0,
          },
        });
      }
    }
  }

  if (results.length === 0) {
    return {
      success: false,
      error: 'All models failed to extract board',
      modelsUsed,
      rawResponses: saveDebugResponses ? rawResponses : undefined,
    };
  }

  // Select best result
  const best = selectBestBoard(results);

  // Verification pass if enabled and confidence below threshold
  if (config.enableVerification && best.confidence < config.confidenceThreshold) {
    onProgress?.({
      step: 'board_verification',
      message: 'Running verification pass...',
      confidence: best.confidence,
    });

    // Use Claude for verification (best at reasoning)
    try {
      const verifyPrompt = VERIFICATION_PROMPT_V2.replace('{rows}', String(best.data.rows))
        .replace('{cols}', String(best.data.cols))
        .replace('{shape}', best.data.shape.replace(/\\n/g, '\n'))
        .replace('{regions}', best.data.regions.replace(/\\n/g, '\n'))
        .replace('{constraints}', JSON.stringify(best.data.constraints, null, 2))
        .replace('{dominoes}', '(not yet extracted)');

      const verifyResponse = await callVisionModel(
        keys,
        'claude-sonnet-4',
        [
          { role: 'system', content: [{ type: 'text', text: CLAUDE_SYSTEM_PROMPT }] },
          {
            role: 'user',
            content: [
              { type: 'image', base64: normalized, mediaType },
              { type: 'text', text: verifyPrompt },
            ],
          },
        ],
        { temperature: 0 }
      );

      modelsUsed.push('claude-sonnet-4');
      const verifyParsed = parseJSONSafely(verifyResponse.text, VerificationSchema);

      if (verifyParsed.success && !verifyParsed.data.verified && verifyParsed.data.corrections) {
        // Apply corrections
        const corrected = { ...best.data };
        if (verifyParsed.data.corrections.rows) corrected.rows = verifyParsed.data.corrections.rows;
        if (verifyParsed.data.corrections.cols) corrected.cols = verifyParsed.data.corrections.cols;
        if (verifyParsed.data.corrections.shape)
          corrected.shape = verifyParsed.data.corrections.shape;
        if (verifyParsed.data.corrections.regions)
          corrected.regions = verifyParsed.data.corrections.regions;
        if (verifyParsed.data.corrections.constraints) {
          corrected.constraints = verifyParsed.data.corrections.constraints;
        }
        return {
          success: true,
          data: corrected,
          modelsUsed,
          rawResponses: saveDebugResponses ? rawResponses : undefined,
          selectedModel: best.model,
        };
      }
    } catch (e) {
      // Verification failed, continue with best result
    }
  }

  return {
    success: true,
    data: best.data,
    modelsUsed,
    rawResponses: saveDebugResponses ? rawResponses : undefined,
    selectedModel: best.model,
  };
}

/**
 * Extract dominoes with ensemble approach
 */
async function extractDominoesEnsemble(
  base64Image: string,
  keys: APIKeys,
  board: BoardExtractionResult,
  strategy: ExtractionStrategy,
  onProgress?: (progress: ExtractionProgress) => void,
  saveDebugResponses?: boolean
): Promise<{
  success: boolean;
  data?: DominoExtractionResult;
  error?: string;
  modelsUsed: string[];
  rawResponses?: DominoModelResponse[];
  selectedModel?: string;
}> {
  const config = STRATEGIES[strategy];
  const modelsUsed: string[] = [];
  const results: DominoResult[] = [];
  const rawResponses: DominoModelResponse[] = [];

  const normalized = normalizeBase64(base64Image);
  const mediaType = inferMediaType(base64Image);

  const modelsToQuery = config.useEnsemble
    ? config.primaryModels.slice(0, config.ensembleSize)
    : [config.primaryModels[0]];

  onProgress?.({
    step: 'dominoes_primary',
    message: `Extracting dominoes with ${modelsToQuery.length} model(s)...`,
    modelsUsed: modelsToQuery,
  });

  // Add board context to prompt
  const contextPrompt = `Board context (already extracted):
- Grid: ${board.rows}x${board.cols}
- Regions: ${Object.keys(board.constraints || {}).length} colored regions

${DOMINO_EXTRACTION_PROMPT_V2}`;

  const queryStartTime = Date.now();
  const responses = await callMultipleModels(
    keys,
    modelsToQuery as (keyof typeof MODELS)[],
    [
      {
        role: 'system',
        content: [{ type: 'text', text: GEMINI_SYSTEM_PROMPT }],
      },
      {
        role: 'user',
        content: [
          { type: 'image', base64: normalized, mediaType },
          { type: 'text', text: contextPrompt },
        ],
      },
    ],
    { temperature: 0.1, jsonMode: true, maxTokens: 8192 }
  );
  const queryEndTime = Date.now();
  const responseMs = queryEndTime - queryStartTime;

  for (const resp of responses) {
    modelsUsed.push(resp.modelKey);

    if (resp.result) {
      const parseStartTime = Date.now();
      const parsed = parseJSONSafely(resp.result.text, DominoExtractionSchema);
      const parseMs = Date.now() - parseStartTime;

      if (parsed.success) {
        const dominoData: DominoExtractionResult = {
          dominoes: parsed.data.dominoes as DominoPair[],
          confidence: parsed.data.confidence || 0.8,
        };

        results.push({
          data: dominoData,
          confidence: parsed.data.confidence || 0.8,
          model: resp.modelKey,
        });

        // Store raw response for debug/comparison
        if (saveDebugResponses) {
          rawResponses.push({
            model: resp.modelKey,
            rawText: resp.result.text,
            parsedData: dominoData,
            parseSuccess: true,
            timing: {
              responseMs: responseMs / responses.length, // Approximate per-model time
              parseMs,
            },
            confidence: parsed.data.confidence || 0.8,
          });
        }
      } else {
        // Store failed parse for debug
        if (saveDebugResponses) {
          rawResponses.push({
            model: resp.modelKey,
            rawText: resp.result.text,
            parsedData: null,
            parseSuccess: false,
            parseError: parsed.error,
            timing: {
              responseMs: responseMs / responses.length,
              parseMs: Date.now() - parseStartTime,
            },
          });
        }
      }
    } else {
      // Store failed call for debug
      if (saveDebugResponses) {
        rawResponses.push({
          model: resp.modelKey,
          rawText: '',
          parsedData: null,
          parseSuccess: false,
          parseError: resp.error || 'Model call failed',
          timing: {
            responseMs: responseMs / responses.length,
            parseMs: 0,
          },
        });
      }
    }
  }

  if (results.length === 0) {
    return {
      success: false,
      error: 'All models failed to extract dominoes',
      modelsUsed,
      rawResponses: saveDebugResponses ? rawResponses : undefined,
    };
  }

  const best = selectBestDominoes(results);
  return {
    success: true,
    data: best.data,
    modelsUsed,
    rawResponses: saveDebugResponses ? rawResponses : undefined,
    selectedModel: best.model,
  };
}

// ════════════════════════════════════════════════════════════════════════════
// Main Export
// ════════════════════════════════════════════════════════════════════════════

export interface EnsembleExtractionResult {
  success: boolean;
  partial?: boolean;
  result?: AIExtractionResult;
  error?: string;
  modelsUsed: string[];
  timing: {
    boardMs: number;
    dominoesMs: number;
    totalMs: number;
  };
  /** Per-model raw responses when saveDebugResponses is enabled */
  rawResponses?: RawResponses;
}

/**
 * Extract puzzle from image using ensemble of models for maximum accuracy
 *
 * @param base64Image - Image for board extraction (may be cropped to puzzle region)
 * @param keys - API keys for models
 * @param options - Extraction options
 * @param dominoImage - Optional image for domino extraction (cropped to domino tray, or full image)
 */
export async function extractPuzzleEnsemble(
  base64Image: string,
  keys: APIKeys,
  options: EnsembleExtractionOptions,
  dominoImage?: string
): Promise<EnsembleExtractionResult> {
  const { strategy, onProgress, saveDebugResponses } = options;
  const startTime = Date.now();
  const allModelsUsed: string[] = [];

  onProgress?.({ step: 'initializing', message: 'Starting extraction...' });

  // Extract board (using potentially cropped image for better accuracy)
  const boardStart = Date.now();
  const boardResult = await extractBoardEnsemble(
    base64Image,
    keys,
    strategy,
    onProgress,
    saveDebugResponses
  );
  const boardMs = Date.now() - boardStart;
  allModelsUsed.push(...boardResult.modelsUsed);

  if (!boardResult.success || !boardResult.data) {
    // Build partial raw responses for failed extraction
    const rawResponses: RawResponses | undefined = saveDebugResponses
      ? {
          board: boardResult.rawResponses || [],
          dominoes: [],
          selectedBoardModel: undefined,
          selectedDominoModel: undefined,
          timing: { boardMs, dominoesMs: 0, totalMs: Date.now() - startTime },
        }
      : undefined;

    return {
      success: false,
      error: boardResult.error || 'Board extraction failed',
      modelsUsed: allModelsUsed,
      timing: { boardMs, dominoesMs: 0, totalMs: Date.now() - startTime },
      rawResponses,
    };
  }

  // Extract dominoes - use dedicated domino image if provided (cropped to tray)
  const imageForDominoes = dominoImage || base64Image;
  const dominoStart = Date.now();
  const dominoResult = await extractDominoesEnsemble(
    imageForDominoes,
    keys,
    boardResult.data,
    strategy,
    onProgress,
    saveDebugResponses
  );
  const dominoesMs = Date.now() - dominoStart;
  allModelsUsed.push(...dominoResult.modelsUsed);

  // Cross-validation if ensemble
  if (STRATEGIES[strategy].useEnsemble) {
    onProgress?.({ step: 'cross_validation', message: 'Cross-validating results...' });
    // Additional cross-validation could go here
  }

  onProgress?.({ step: 'complete', message: 'Extraction complete' });

  // Build raw responses if debug mode enabled
  const totalMs = Date.now() - startTime;
  const rawResponses: RawResponses | undefined = saveDebugResponses
    ? {
        board: boardResult.rawResponses || [],
        dominoes: dominoResult.rawResponses || [],
        selectedBoardModel: boardResult.selectedModel,
        selectedDominoModel: dominoResult.selectedModel,
        timing: { boardMs, dominoesMs, totalMs },
      }
    : undefined;

  if (!dominoResult.success || !dominoResult.data) {
    // Partial success - board extracted, dominoes failed
    return {
      success: true,
      partial: true,
      result: {
        board: boardResult.data,
        dominoes: { dominoes: [], confidence: 0 },
        reasoning: `Board extracted successfully. Dominoes: FAILED - ${dominoResult.error}`,
        debug: rawResponses ? { rawResponses } : undefined,
      },
      error: dominoResult.error,
      modelsUsed: allModelsUsed,
      timing: { boardMs, dominoesMs, totalMs },
      rawResponses,
    };
  }

  return {
    success: true,
    partial: false,
    result: {
      board: boardResult.data,
      dominoes: dominoResult.data,
      reasoning: `Ensemble extraction using ${[...new Set(allModelsUsed)].join(', ')}`,
      debug: rawResponses ? { rawResponses } : undefined,
    },
    modelsUsed: allModelsUsed,
    timing: { boardMs, dominoesMs, totalMs },
    rawResponses,
  };
}

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
  DominoExtractionResult,
  DominoPair,
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
const BOARD_EXTRACTION_PROMPT_V2 = `Analyze this NYT Pips puzzle screenshot. Extract the grid structure.

TASK 1: GRID DIMENSIONS
- Count rows (horizontal lines of cells, top to bottom)
- Count columns (cells per row, left to right)
- Note: Some corners may have MISSING cells (holes)

TASK 2: GRID LOCATION (pixel coordinates)
Identify exact pixel boundaries:
- left: leftmost cell edge X
- top: topmost cell edge Y
- right: rightmost cell edge X
- bottom: bottommost cell edge Y
Also provide full imageWidth and imageHeight.

TASK 3: SHAPE MAP
For each cell position, mark:
- "." if cell EXISTS
- "#" if cell is MISSING (hole)
Join rows with \\n

TASK 4: REGION MAP
For each existing cell, identify its colored region:
- Use letters A-J for different colors
- Use "#" for holes
- Use "." if no color visible
Join rows with \\n

TASK 5: CONSTRAINTS
Read numbers/symbols near colored regions:
- "=12" or "12" → sum equals 12
- "<10" → sum less than 10
- ">5" → sum greater than 5
- "≠" or "all different" → all_different constraint
- "=" with no number → all_equal constraint

OUTPUT (JSON only, no markdown):
{
  "rows": 5,
  "cols": 5,
  "gridLocation": {
    "left": 95,
    "top": 611,
    "right": 625,
    "bottom": 1141,
    "imageWidth": 720,
    "imageHeight": 1560
  },
  "shape": "....#\\n.....\\n.....\\n.....\\n#....",
  "regions": "AAAB#\\nAABBC\\nDDBCC\\nDDEEE\\n#FEEE",
  "constraints": {
    "A": {"type": "sum", "op": "==", "value": 8},
    "B": {"type": "sum", "op": "<", "value": 12}
  },
  "confidence": {"grid": 0.95, "regions": 0.88, "constraints": 0.92},
  "reasoning": "5x5 grid with holes at corners..."
}`;

/**
 * Optimized domino extraction prompt
 */
const DOMINO_EXTRACTION_PROMPT_V2 = `Analyze the DOMINO TRAY in this Pips puzzle screenshot.

The dominoes are shown in a separate area (usually below or beside the main grid).
Each domino has TWO halves, each with 0-6 pips (dots).

COUNTING GUIDE:
- 0 pips: Blank/empty
- 1 pip: Single center dot
- 2 pips: Diagonal pair
- 3 pips: Diagonal line
- 4 pips: Four corners
- 5 pips: Four corners + center
- 6 pips: Two columns of 3

Count EACH domino carefully. Scan left-to-right, top-to-bottom.

OUTPUT (JSON only):
{
  "dominoes": [[6, 4], [5, 3], [2, 1], [0, 0]],
  "confidence": 0.92,
  "reasoning": "Found 4 dominoes in tray. Clear pip counts."
}

IMPORTANT:
- Only count dominoes in the TRAY area, not on the grid
- Each domino is [leftPips, rightPips] or [topPips, bottomPips]
- Double-check your pip counts
- Set confidence < 0.8 if any pips are unclear`;

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

const ConstraintSchema = z.object({
  type: z.enum(['sum', 'all_equal', 'all_different']),
  op: z.enum(['==', '<', '>', '!=']).optional(),
  value: z.number().optional(),
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
}

/**
 * Extract board structure with ensemble approach
 */
async function extractBoardEnsemble(
  base64Image: string,
  keys: APIKeys,
  strategy: ExtractionStrategy,
  onProgress?: (progress: ExtractionProgress) => void
): Promise<{
  success: boolean;
  data?: BoardExtractionResult;
  error?: string;
  modelsUsed: string[];
}> {
  const config = STRATEGIES[strategy];
  const modelsUsed: string[] = [];
  const results: BoardResult[] = [];

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
    { temperature: 0.1, jsonMode: true }
  );

  // Process results
  for (const resp of responses) {
    modelsUsed.push(resp.modelKey);
    if (resp.result) {
      const parsed = parseJSONSafely(resp.result.text, BoardExtractionSchema);
      if (parsed.success) {
        const confidence = parsed.data.confidence
          ? (parsed.data.confidence.grid +
              parsed.data.confidence.regions +
              parsed.data.confidence.constraints) /
            3
          : 0.8;
        results.push({
          data: {
            rows: parsed.data.rows,
            cols: parsed.data.cols,
            shape: parsed.data.shape,
            regions: parsed.data.regions,
            constraints: parsed.data.constraints || {},
            confidence: parsed.data.confidence || { grid: 0.8, regions: 0.8, constraints: 0.8 },
            gridLocation: parsed.data.gridLocation,
          },
          confidence,
          model: resp.modelKey,
        });
      } else {
        console.warn(`[Ensemble] ${resp.modelKey} parse failed:`, parsed.error);
      }
    } else {
      console.warn(`[Ensemble] ${resp.modelKey} call failed:`, resp.error);
    }
  }

  if (results.length === 0) {
    return { success: false, error: 'All models failed to extract board', modelsUsed };
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
        console.log('[Ensemble] Verification found issues:', verifyParsed.data.issues);
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
        return { success: true, data: corrected, modelsUsed };
      }
    } catch (e) {
      console.warn('[Ensemble] Verification failed:', e);
    }
  }

  return { success: true, data: best.data, modelsUsed };
}

/**
 * Extract dominoes with ensemble approach
 */
async function extractDominoesEnsemble(
  base64Image: string,
  keys: APIKeys,
  board: BoardExtractionResult,
  strategy: ExtractionStrategy,
  onProgress?: (progress: ExtractionProgress) => void
): Promise<{
  success: boolean;
  data?: DominoExtractionResult;
  error?: string;
  modelsUsed: string[];
}> {
  const config = STRATEGIES[strategy];
  const modelsUsed: string[] = [];
  const results: DominoResult[] = [];

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
    { temperature: 0.1, jsonMode: true }
  );

  for (const resp of responses) {
    modelsUsed.push(resp.modelKey);
    if (resp.result) {
      const parsed = parseJSONSafely(resp.result.text, DominoExtractionSchema);
      if (parsed.success) {
        results.push({
          data: {
            dominoes: parsed.data.dominoes as DominoPair[],
            confidence: parsed.data.confidence || 0.8,
          },
          confidence: parsed.data.confidence || 0.8,
          model: resp.modelKey,
        });
      } else {
        console.warn(`[Ensemble] ${resp.modelKey} domino parse failed:`, parsed.error);
      }
    } else {
      console.warn(`[Ensemble] ${resp.modelKey} domino call failed:`, resp.error);
    }
  }

  if (results.length === 0) {
    return { success: false, error: 'All models failed to extract dominoes', modelsUsed };
  }

  const best = selectBestDominoes(results);
  return { success: true, data: best.data, modelsUsed };
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
}

/**
 * Extract puzzle from image using ensemble of models for maximum accuracy
 */
export async function extractPuzzleEnsemble(
  base64Image: string,
  keys: APIKeys,
  options: EnsembleExtractionOptions
): Promise<EnsembleExtractionResult> {
  const { strategy, onProgress } = options;
  const startTime = Date.now();
  const allModelsUsed: string[] = [];

  onProgress?.({ step: 'initializing', message: 'Starting extraction...' });

  // Extract board
  const boardStart = Date.now();
  const boardResult = await extractBoardEnsemble(base64Image, keys, strategy, onProgress);
  const boardMs = Date.now() - boardStart;
  allModelsUsed.push(...boardResult.modelsUsed);

  if (!boardResult.success || !boardResult.data) {
    return {
      success: false,
      error: boardResult.error || 'Board extraction failed',
      modelsUsed: allModelsUsed,
      timing: { boardMs, dominoesMs: 0, totalMs: Date.now() - startTime },
    };
  }

  // Extract dominoes
  const dominoStart = Date.now();
  const dominoResult = await extractDominoesEnsemble(
    base64Image,
    keys,
    boardResult.data,
    strategy,
    onProgress
  );
  const dominoesMs = Date.now() - dominoStart;
  allModelsUsed.push(...dominoResult.modelsUsed);

  // Cross-validation if ensemble
  if (STRATEGIES[strategy].useEnsemble) {
    onProgress?.({ step: 'cross_validation', message: 'Cross-validating results...' });
    // Additional cross-validation could go here
  }

  onProgress?.({ step: 'complete', message: 'Extraction complete' });

  if (!dominoResult.success || !dominoResult.data) {
    // Partial success - board extracted, dominoes failed
    return {
      success: true,
      partial: true,
      result: {
        board: boardResult.data,
        dominoes: { dominoes: [], confidence: 0 },
        reasoning: `Board extracted successfully. Dominoes: FAILED - ${dominoResult.error}`,
      },
      error: dominoResult.error,
      modelsUsed: allModelsUsed,
      timing: { boardMs, dominoesMs, totalMs: Date.now() - startTime },
    };
  }

  return {
    success: true,
    partial: false,
    result: {
      board: boardResult.data,
      dominoes: dominoResult.data,
      reasoning: `Ensemble extraction using ${[...new Set(allModelsUsed)].join(', ')}`,
    },
    modelsUsed: allModelsUsed,
    timing: { boardMs, dominoesMs, totalMs: Date.now() - startTime },
  };
}

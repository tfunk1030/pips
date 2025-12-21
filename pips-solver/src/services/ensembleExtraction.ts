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

import { APIKeys, ExtractionStrategy, MODELS, STRATEGIES } from '../config/models';
import {
  AIExtractionResult,
  BoardExtractionResult,
  DominoExtractionResult,
  DominoPair,
} from '../model/overlayTypes';
import {
  BoardExtractionSchema,
  DominoExtractionSchema,
  VerificationSchema,
} from './extractionSchemas';
import { parseJSONSafely } from './jsonParsingUtils';
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
const BOARD_EXTRACTION_PROMPT_V2 = `Analyze this Pips puzzle image and extract its structure.

STEP-BY-STEP EXTRACTION METHOD:

═══════════════════════════════════════════════════════════════════════════════
STEP 1: COUNT GRID DIMENSIONS (CRITICAL - COUNT VERY CAREFULLY!)
═══════════════════════════════════════════════════════════════════════════════
The puzzle grid has VISIBLE BLACK GRIDLINES. Use them to count precisely:

COUNTING METHOD - Use gridlines:
1. Count all VERTICAL black lines from left to right edge of the grid
2. Number of columns = (vertical lines) - 1
3. Count all HORIZONTAL black lines from top to bottom edge of the grid
4. Number of rows = (horizontal lines) - 1

CRITICAL: Do NOT count the domino tray area (usually below the grid) - only the main puzzle grid!

EXAMPLE: If you see 5 vertical lines and 4 horizontal lines → grid is 4 columns × 3 rows

COMMON MISTAKE TO AVOID:
- Don't just count colored cells! Count the FULL grid including holes/gaps
- Holes look like dark/empty squares but they're INSIDE the grid boundary
- The grid boundary is defined by the outermost gridlines
- Do NOT include the domino tray in your row count

Typical NYT Pips grid sizes: 4×3, 4×4, 5×4, 5×5, 6×4, 6×5, 6×6

═══════════════════════════════════════════════════════════════════════════════
STEP 2: BUILD THE SHAPE STRING (cell existence map)
═══════════════════════════════════════════════════════════════════════════════
Go ROW BY ROW, from top to bottom. For each position in the bounding box:
- "." = A cell EXISTS here (has a colored background, can hold a domino half)
- "#" = NO cell here (empty space, hole, gap, dark square)

CRITICAL: The shape field uses ONLY "." and "#" characters!
NEVER put region letters (A, B, C) in the shape field!

Join rows with literal \\n (backslash-n, not actual newlines).

SELF-CHECK:
- shape must have exactly [rows] lines
- each line must have exactly [cols] characters
- total cells (dots) + total holes (hashes) = rows × cols

═══════════════════════════════════════════════════════════════════════════════
STEP 3: BUILD THE REGIONS STRING (color map)
═══════════════════════════════════════════════════════════════════════════════
Go ROW BY ROW again, same order. For each position:
- Assign letters A, B, C, D... to distinct BACKGROUND COLORS (not border colors)
- Use "#" for holes (must EXACTLY match the shape field)
- Same color = same letter throughout the entire grid

Common region colors: pink, coral, orange, peach/tan, teal/cyan, gray, olive, green, purple

Join rows with literal \\n. The regions string must have IDENTICAL dimensions to shape.

═══════════════════════════════════════════════════════════════════════════════
STEP 4: EXTRACT CONSTRAINTS
═══════════════════════════════════════════════════════════════════════════════
Look for diamond-shaped labels on or near each colored region:
- Number (e.g., "8", "12") → {"type": "sum", "op": "==", "value": N}
- "0" → {"type": "sum", "op": "==", "value": 0}
- "=" symbol alone → {"type": "all_equal"}
- "<N" → {"type": "sum", "op": "<", "value": N}
- ">N" → {"type": "sum", "op": ">", "value": N}

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only - no markdown, no explanation)
═══════════════════════════════════════════════════════════════════════════════
{
  "rows": <number of rows in bounding box>,
  "cols": <number of columns in bounding box>,
  "shape": "<dots and hashes joined by \\n>",
  "regions": "<letters and hashes joined by \\n>",
  "constraints": {"A": {...}, "B": {...}},
  "confidence": {"grid": 0.0-1.0, "regions": 0.0-1.0, "constraints": 0.0-1.0},
  "reasoning": "<brief description of what you found>"
}`;

/**
 * Optimized domino extraction prompt
 */
const DOMINO_EXTRACTION_PROMPT_V2 = `Extract ALL dominoes from this Pips puzzle image.

═══════════════════════════════════════════════════════════════════════════════
DOMINO LOCATION
═══════════════════════════════════════════════════════════════════════════════
Dominoes are in a TRAY/BANK area, separate from the main puzzle grid.
Usually located BELOW the puzzle. May span multiple rows.

═══════════════════════════════════════════════════════════════════════════════
PIP COUNTING PATTERNS
═══════════════════════════════════════════════════════════════════════════════
Each domino half shows 0-6 pips (dots):
- 0: Blank (no dots)
- 1: One dot in center
- 2: Two dots diagonal
- 3: Three dots diagonal line
- 4: Four dots in corners
- 5: Four corners + center
- 6: Two columns of 3 dots (6 total)

═══════════════════════════════════════════════════════════════════════════════
EXTRACTION METHOD
═══════════════════════════════════════════════════════════════════════════════
1. Locate the domino tray (below or beside the puzzle grid)
2. Scan LEFT to RIGHT, TOP to BOTTOM - cover ALL rows of the tray
3. For each domino: count pips on BOTH halves
4. Record as [first_half, second_half]
5. Continue until you've captured EVERY visible domino

IMPORTANT: Count ALL dominoes! The number varies by puzzle size:
- Small puzzles: 7-8 dominoes
- Medium puzzles: 9-12 dominoes
- Large puzzles: 12-14+ dominoes
Do NOT assume a fixed count - extract what you SEE.

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only - no markdown)
═══════════════════════════════════════════════════════════════════════════════
{
  "dominoes": [[a, b], [c, d], ...],
  "confidence": 0.0-1.0,
  "reasoning": "<how many dominoes found, any issues>"
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

═══════════════════════════════════════════════════════════════════════════════
VERIFICATION CHECKLIST (check each carefully):
═══════════════════════════════════════════════════════════════════════════════

1. GRID DIMENSIONS (most common error!):
   - Count the BLACK GRIDLINES in the image
   - Columns = (vertical lines) - 1
   - Rows = (horizontal lines) - 1
   - Does {rows}x{cols} match your count?

2. SHAPE STRING:
   - Does it have exactly {rows} lines?
   - Does each line have exactly {cols} characters?
   - Are holes (#) in the right positions?

3. REGIONS:
   - Same dimensions as shape?
   - Each color mapped to a consistent letter?
   - Holes (#) match exactly?

4. CONSTRAINTS:
   - Every region with a diamond label captured?
   - Values correct?

5. DOMINOES (if extracted):
   - All dominoes from the tray captured?
   - Pip counts accurate?

OUTPUT (JSON only):
{
  "verified": true,
  "issues": [],
  "corrections": null
}

OR if corrections needed:
{
  "verified": false,
  "issues": ["Grid should be 4x3 not 3x3", "Missing column in shape"],
  "corrections": {
    "rows": <corrected row count if wrong>,
    "cols": <corrected col count if wrong>,
    "shape": "corrected shape if wrong",
    "regions": "corrected regions if wrong",
    "dominoes": [[a, b], ...],
    "constraints": { ... }
  }
}`;

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
 * Result from board selection with consensus info
 */
interface BoardSelectionResult {
  best: BoardResult;
  consensusScore: number;
  combinedScore: number;
}

/**
 * Select best result from ensemble based on confidence and consensus
 */
function selectBestBoard(results: BoardResult[]): BoardSelectionResult {
  if (results.length === 1) {
    return { best: results[0], consensusScore: 1.0, combinedScore: results[0].confidence };
  }

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

  return {
    best: scored[0].result,
    consensusScore: scored[0].consensusScore,
    combinedScore: scored[0].combinedScore,
  };
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
    { temperature: 0.1, jsonMode: true, maxTokens: 16384 }
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
  const selection = selectBestBoard(results);
  const best = selection.best;

  // Trigger verification if:
  // 1. Verification is enabled AND
  // 2. Either confidence is below threshold OR consensus is low (models disagree)
  const LOW_CONSENSUS_THRESHOLD = 0.5;
  const needsVerification =
    config.enableVerification &&
    (selection.combinedScore < config.confidenceThreshold ||
      selection.consensusScore < LOW_CONSENSUS_THRESHOLD);

  if (needsVerification) {
    const reason =
      selection.consensusScore < LOW_CONSENSUS_THRESHOLD
        ? `low consensus (${(selection.consensusScore * 100).toFixed(0)}%)`
        : `low confidence (${(selection.combinedScore * 100).toFixed(0)}%)`;

    console.log(`[Ensemble] Running verification due to ${reason}`);
    onProgress?.({
      step: 'board_verification',
      message: `Running verification (${reason})...`,
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
      } else if (verifyParsed.success && verifyParsed.data.verified) {
        console.log('[Ensemble] Verification confirmed extraction is correct');
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
    { temperature: 0.1, jsonMode: true, maxTokens: 8192 }
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
  const { strategy, onProgress } = options;
  const startTime = Date.now();
  const allModelsUsed: string[] = [];

  onProgress?.({ step: 'initializing', message: 'Starting extraction...' });

  // Extract board (using potentially cropped image for better accuracy)
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

  // Extract dominoes - use dedicated domino image if provided (cropped to tray)
  const imageForDominoes = dominoImage || base64Image;
  const dominoStart = Date.now();
  const dominoResult = await extractDominoesEnsemble(
    imageForDominoes,
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

/**
 * AI Extraction Service
 * Multi-model extraction with ensemble support for maximum accuracy
 *
 * Supports:
 * - Gemini 2.5 Pro (best for grid/object detection, mAP 13.3)
 * - Claude Sonnet 4 (best for structured JSON output, 85% accuracy)
 * - GPT-4o (fallback)
 *
 * Strategies:
 * - fast: Single model (Gemini Flash), ~3s
 * - balanced: Gemini Pro + verification, ~20s
 * - accurate: Gemini Pro + Claude verification, ~35s
 * - ensemble: Multi-model consensus voting, ~45s (MAXIMUM ACCURACY)
 */

import { Platform } from 'react-native';
import { z } from 'zod';
import { ExtractionStrategy, MODEL_CANDIDATES, getAvailableProviders } from '../config/models';
import {
  AIExtractionResult,
  BoardExtractionResult,
  ConstraintDef,
  ConstraintState,
  DEFAULT_GRID_BOUNDS,
  DEFAULT_PALETTE,
  DominoExtractionResult,
  DominoPair,
  GridState,
  RegionState,
} from '../model/overlayTypes';
import {
  ExtractionProgress as EnsembleProgress,
  extractPuzzleEnsemble,
} from './ensembleExtraction';
import { APIKeys } from './modelClients';

// ════════════════════════════════════════════════════════════════════════════
// Zod Schemas for AI Response Validation
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

const ConfidenceScoresSchema = z
  .object({
    grid: z.number().min(0).max(1),
    regions: z.number().min(0).max(1),
    constraints: z.number().min(0).max(1),
  })
  .optional();

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
  confidence: ConfidenceScoresSchema,
  reasoning: z.string().optional(),
});

const DominoExtractionSchema = z.object({
  dominoes: z.array(z.tuple([z.number().min(0).max(6), z.number().min(0).max(6)])),
  confidence: z.number().min(0).max(1).optional(),
  reasoning: z.string().optional(),
});

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

interface ClaudeMessage {
  role: 'user' | 'assistant';
  content: ClaudeContent[];
}

type ClaudeContent =
  | { type: 'text'; text: string }
  | { type: 'image'; source: { type: 'base64'; media_type: string; data: string } };

interface ClaudeResponse {
  content: Array<{ type: 'text'; text: string }>;
  stop_reason: string;
}

export interface ExtractionProgress {
  step: 'board' | 'dominoes' | 'done';
  message: string;
}

// ════════════════════════════════════════════════════════════════════════════
// Multi-Model Extraction (Maximum Accuracy)
// ════════════════════════════════════════════════════════════════════════════

export interface MultiModelExtractionOptions {
  /**
   * Extraction strategy:
   * - 'fast': Single model (Gemini Flash), ~3s, good accuracy
   * - 'balanced': Gemini Pro with verification, ~20s, high accuracy
   * - 'accurate': Gemini Pro + Claude verification, ~35s, very high accuracy
   * - 'ensemble': Multi-model consensus, ~45s, MAXIMUM ACCURACY
   */
  strategy: ExtractionStrategy;

  /** API keys for different providers */
  apiKeys: APIKeys;

  /** Progress callback */
  onProgress?: (progress: EnsembleProgress) => void;

  /**
   * Use hybrid CV + AI extraction (recommended for best accuracy)
   * When enabled:
   * 1. CV service crops image to puzzle region only (faster, cleaner)
   * 2. AI analyzes the cropped image (more accurate)
   *
   * Requires CV service running at cvServiceUrl
   */
  useHybridCV?: boolean;

  /** CV service URL (default: http://localhost:8080) */
  cvServiceUrl?: string;
}

export interface MultiModelExtractionResult {
  success: boolean;
  partial?: boolean;
  result?: AIExtractionResult;
  error?: string;
  /** Models that were used in the extraction */
  modelsUsed: string[];
  /** Timing breakdown in milliseconds */
  timing: {
    boardMs: number;
    dominoesMs: number;
    totalMs: number;
  };
}

/**
 * Extract puzzle from screenshot using multi-model ensemble for maximum accuracy
 *
 * This is the recommended extraction method for production use.
 *
 * @param base64Image - Base64 encoded image data
 * @param options - Extraction options including strategy and API keys
 * @returns Extraction result with timing and model provenance
 *
 * @example
 * ```typescript
 * const result = await extractPuzzleMultiModel(imageBase64, {
 *   strategy: 'ensemble', // Maximum accuracy
 *   apiKeys: {
 *     google: process.env.GOOGLE_API_KEY,
 *     anthropic: process.env.ANTHROPIC_API_KEY,
 *   },
 *   onProgress: (p) => console.log(p.message),
 * });
 * ```
 */
export async function extractPuzzleMultiModel(
  base64Image: string,
  options: MultiModelExtractionOptions
): Promise<MultiModelExtractionResult> {
  const { strategy, apiKeys, onProgress, useHybridCV = false, cvServiceUrl } = options;

  // Validate we have at least one API key
  const availableProviders = getAvailableProviders(apiKeys);
  if (availableProviders.length === 0) {
    return {
      success: false,
      error:
        'No API keys provided. At least one of: google, anthropic, or openai API key is required.',
      modelsUsed: [],
      timing: { boardMs: 0, dominoesMs: 0, totalMs: 0 },
    };
  }

  // Log available providers
  console.log(`[MultiModel] Available providers: ${availableProviders.join(', ')}`);
  console.log(`[MultiModel] Strategy: ${strategy}`);
  console.log(`[MultiModel] Hybrid CV: ${useHybridCV ? 'enabled' : 'disabled'}`);

  let imageForAI = base64Image;
  let cvCropMs = 0;

  // If hybrid mode is enabled, try to crop puzzle region first
  // We keep both full image (for dominoes) and cropped image (for board)
  let usedCropping = false;
  let cropBounds:
    | {
        x: number;
        y: number;
        width: number;
        height: number;
        originalWidth: number;
        originalHeight: number;
      }
    | undefined;

  // Actual grid bounds (without padding) for overlay alignment
  let gridBoundsForOverlay:
    | {
        x: number;
        y: number;
        width: number;
        height: number;
        originalWidth: number;
        originalHeight: number;
      }
    | undefined;

  let dominoImage: string | undefined;

  if (useHybridCV) {
    try {
      // Import CV extraction dynamically to avoid circular deps
      const { cropPuzzleRegion, cropDominoRegion, setCVServiceURL } = await import(
        './cvExtraction'
      );

      // Set CV service URL if provided
      if (cvServiceUrl) {
        setCVServiceURL(cvServiceUrl);
      }

      onProgress?.({
        step: 'initializing',
        message: 'Using CV to crop puzzle region...',
        confidence: 0,
      });

      // Crop puzzle region for board extraction
      const cropResult = await cropPuzzleRegion(base64Image);
      cvCropMs = cropResult.extractionMs;

      if (cropResult.success && cropResult.croppedImage && cropResult.bounds) {
        console.log(
          `[MultiModel] Puzzle crop successful: ${cropResult.bounds.width}x${cropResult.bounds.height} in ${cvCropMs}ms`
        );
        imageForAI = cropResult.croppedImage;
        usedCropping = true;
        cropBounds = cropResult.bounds;
        gridBoundsForOverlay = cropResult.gridBounds; // Use actual grid bounds for overlay

        onProgress?.({
          step: 'initializing',
          message: `Cropped puzzle region (${cropResult.bounds.width}x${cropResult.bounds.height})`,
          confidence: 0.1,
        });

        // Crop domino region for domino extraction
        // Use the puzzle bottom Y coordinate to know where dominoes start
        const puzzleBottomY = cropResult.bounds.y + cropResult.bounds.height;
        const dominoCropResult = await cropDominoRegion(base64Image, puzzleBottomY);
        cvCropMs += dominoCropResult.extractionMs;

        if (dominoCropResult.success && dominoCropResult.croppedImage) {
          console.log(
            `[MultiModel] Domino crop successful: ${dominoCropResult.bounds?.width}x${dominoCropResult.bounds?.height}`
          );
          dominoImage = dominoCropResult.croppedImage;

          onProgress?.({
            step: 'initializing',
            message: `Cropped domino region (${dominoCropResult.bounds?.width}x${dominoCropResult.bounds?.height})`,
            confidence: 0.15,
          });
        } else {
          console.warn(
            `[MultiModel] Domino crop failed: ${dominoCropResult.error}, using full image`
          );
          dominoImage = base64Image; // Fallback to full image
        }
      } else {
        console.warn(`[MultiModel] CV crop failed: ${cropResult.error}, using full image`);
        // Continue with full image if crop fails
      }
    } catch (error) {
      console.warn('[MultiModel] CV service unavailable, using full image:', error);
      // Continue with full image if CV service is unavailable
    }
  }

  // Use ensemble extraction:
  // - Cropped puzzle image for board extraction (better accuracy on grid/regions)
  // - Cropped domino image for domino extraction (focused on just the dominoes)
  const result = await extractPuzzleEnsemble(
    imageForAI,
    apiKeys,
    { strategy, onProgress },
    dominoImage // Use cropped domino image if available
  );

  // Add CV crop time to total timing
  if (cvCropMs > 0 && result.timing) {
    result.timing.totalMs += cvCropMs;
  }

  // Use CV's actual grid bounds (without padding) for precise overlay alignment
  if (usedCropping && gridBoundsForOverlay && result.success && result.result?.board) {
    const cvBasedLocation = {
      left: gridBoundsForOverlay.x,
      top: gridBoundsForOverlay.y,
      right: gridBoundsForOverlay.x + gridBoundsForOverlay.width,
      bottom: gridBoundsForOverlay.y + gridBoundsForOverlay.height,
      imageWidth: gridBoundsForOverlay.originalWidth,
      imageHeight: gridBoundsForOverlay.originalHeight,
    };

    console.log(
      `[MultiModel] Using CV grid bounds for overlay: (${cvBasedLocation.left},${cvBasedLocation.top})-(${cvBasedLocation.right},${cvBasedLocation.bottom})`
    );

    result.result.board.gridLocation = cvBasedLocation;
  }

  return result;
}

/**
 * Convenience function for maximum accuracy extraction
 * Uses ensemble strategy with all available models + hybrid CV preprocessing
 *
 * @param useHybridCV - Enable CV preprocessing (requires CV service running)
 */
export async function extractPuzzleMaxAccuracy(
  base64Image: string,
  apiKeys: APIKeys,
  onProgress?: (progress: EnsembleProgress) => void,
  useHybridCV: boolean = false
): Promise<MultiModelExtractionResult> {
  return extractPuzzleMultiModel(base64Image, {
    strategy: 'ensemble',
    apiKeys,
    onProgress,
    useHybridCV,
  });
}

function normalizeBase64ImageData(input: string): string {
  let s = input.trim();

  // Handle "data:image/...;base64,xxxx" strings.
  if (s.startsWith('data:')) {
    const idx = s.indexOf('base64,');
    if (idx >= 0) {
      s = s.slice(idx + 'base64,'.length);
    }
  }

  // Remove any whitespace/newlines that may break decoding.
  s = s.replace(/\s+/g, '');

  // Note: We don't add padding here because base64 padding is part of the encoding
  // and should already be present if needed. Adding padding could corrupt valid data.

  return s;
}

function inferImageMediaTypeFromBase64(base64: string): string {
  const b64 = normalizeBase64ImageData(base64);

  // Try to decode first few bytes to check magic numbers
  let detectedType = 'unknown';
  if (typeof atob !== 'undefined') {
    try {
      // Decode first 16 base64 chars (12 bytes) - enough for all image format signatures
      // Ensure we have enough characters and pad if needed
      let sample = b64.substring(0, 16);
      // Add padding if needed for valid base64
      while (sample.length % 4 !== 0) {
        sample += '=';
      }
      const firstBytes = atob(sample);
      const bytes = new Uint8Array(firstBytes.length);
      for (let i = 0; i < firstBytes.length; i++) {
        bytes[i] = firstBytes.charCodeAt(i);
      }

      // PNG: 89 50 4E 47 0D 0A 1A 0A
      if (
        bytes.length >= 8 &&
        bytes[0] === 0x89 &&
        bytes[1] === 0x50 &&
        bytes[2] === 0x4e &&
        bytes[3] === 0x47
      ) {
        detectedType = 'PNG';
      }
      // JPEG: FF D8 FF
      else if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
        detectedType = 'JPEG';
      }
      // WebP: RIFF...WEBP (bytes 0-3: 'RIFF', bytes 8-11: 'WEBP')
      else if (
        bytes.length >= 12 &&
        bytes[0] === 0x52 &&
        bytes[1] === 0x49 &&
        bytes[2] === 0x46 &&
        bytes[3] === 0x46 &&
        bytes[8] === 0x57 &&
        bytes[9] === 0x45 &&
        bytes[10] === 0x42 &&
        bytes[11] === 0x50
      ) {
        detectedType = 'WebP';
      }
    } catch (e) {
      // Fall back to string-based detection if decoding fails
      console.warn('[DEBUG] Failed to decode base64 for type detection:', e);
    }
  }

  // Fallback to string-based detection if binary detection didn't work
  if (detectedType === 'unknown') {
    // PNG signature: 89 50 4E 47 0D 0A 1A 0A -> iVBORw0KGgo
    if (b64.startsWith('iVBORw0KGgo')) {
      detectedType = 'PNG';
    }
    // JPEG signature: FF D8 FF -> /9j/
    else if (b64.startsWith('/9j/')) {
      detectedType = 'JPEG';
    }
    // WebP "RIFF" header -> UklGR
    else if (b64.startsWith('UklGR')) {
      detectedType = 'WebP';
    }
  }

  const mediaTypeMap: Record<string, string> = {
    PNG: 'image/png',
    JPEG: 'image/jpeg',
    WebP: 'image/webp',
  };

  const mediaType = mediaTypeMap[detectedType] || 'image/jpeg'; // Default to JPEG

  return mediaType;
}

// ════════════════════════════════════════════════════════════════════════════
// Main Extraction Function
// ════════════════════════════════════════════════════════════════════════════

/**
 * Extract puzzle data from a screenshot using Claude's vision
 * Uses a two-pass approach: first extracts board, then extracts dominoes
 * Supports partial success - board results are returned even if dominoes fail
 * Optional verification pass can be enabled for self-correction
 */
export async function extractPuzzleFromImage(
  base64Image: string,
  apiKey: string,
  onProgress?: (progress: ExtractionProgress) => void,
  enableVerification: boolean = false
): Promise<{
  success: boolean;
  partial?: boolean;
  result?: AIExtractionResult;
  error?: string;
}> {
  const trimmedKey = apiKey.trim();
  if (!trimmedKey) {
    return { success: false, error: 'API key is required' };
  }

  const normalizedImage = normalizeBase64ImageData(base64Image);
  if (!normalizedImage) {
    return { success: false, error: 'Image data is empty' };
  }

  try {
    // Pass 1: Extract board structure
    console.log('[DEBUG] extractPuzzleFromImage - Starting board extraction...');
    onProgress?.({ step: 'board', message: 'Analyzing board structure...' });
    const boardExtractStartTime = Date.now();
    const boardResult = await extractBoard(normalizedImage, trimmedKey);
    const boardExtractDuration = Date.now() - boardExtractStartTime;
    console.log(
      `[DEBUG] extractPuzzleFromImage - Board extraction completed in ${boardExtractDuration}ms, success: ${boardResult.success}`
    );

    if (!boardResult.success || !boardResult.data) {
      return { success: false, error: boardResult.error || 'Failed to extract board' };
    }

    // Optional: Verification pass to check extraction accuracy
    let finalBoardData = boardResult.data;
    if (enableVerification && boardResult.data.confidence) {
      const avgConfidence =
        (boardResult.data.confidence.grid +
          boardResult.data.confidence.regions +
          boardResult.data.confidence.constraints) /
        3;

      // Only verify if confidence is below 90%
      if (avgConfidence < 0.9) {
        console.log('[DEBUG] Low confidence detected, running verification pass...');
        onProgress?.({ step: 'board', message: 'Verifying extraction...' });

        const verificationResult = await verifyBoardExtraction(
          normalizedImage,
          trimmedKey,
          boardResult.data
        );

        if (verificationResult.success && verificationResult.corrections) {
          console.log('[DEBUG] Verification suggested corrections, applying...');
          finalBoardData = verificationResult.corrections;
        }
      }
    }

    // Pass 2: Extract dominoes (independent - partial success allowed)
    console.log('[DEBUG] extractPuzzleFromImage - Starting domino extraction...');
    onProgress?.({ step: 'dominoes', message: 'Extracting dominoes...' });
    const dominoExtractStartTime = Date.now();
    const dominoResult = await extractDominoes(normalizedImage, trimmedKey, finalBoardData);
    const dominoExtractDuration = Date.now() - dominoExtractStartTime;
    console.log(
      `[DEBUG] extractPuzzleFromImage - Domino extraction completed in ${dominoExtractDuration}ms, success: ${dominoResult.success}`
    );

    // If dominoes fail, return partial success with board data
    if (!dominoResult.success || !dominoResult.data) {
      onProgress?.({ step: 'done', message: 'Extraction partial - dominoes failed' });
      return {
        success: true,
        partial: true,
        result: {
          board: finalBoardData,
          dominoes: { dominoes: [] }, // Empty dominoes - user can add manually
          reasoning: `Board: ${boardResult.reasoning}\n\nDominoes: FAILED - ${
            dominoResult.error || 'Unknown error'
          }. Please add dominoes manually.`,
        },
        error: `Board extracted successfully, but domino extraction failed: ${dominoResult.error}`,
      };
    }

    onProgress?.({ step: 'done', message: 'Extraction complete' });

    return {
      success: true,
      partial: false,
      result: {
        board: finalBoardData,
        dominoes: dominoResult.data,
        reasoning: `Board: ${boardResult.reasoning}\n\nDominoes: ${dominoResult.reasoning}`,
      },
    };
  } catch (error) {
    return {
      success: false,
      error: `Extraction failed: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Verification Pass: Self-Correction
// ════════════════════════════════════════════════════════════════════════════

const VERIFICATION_PROMPT = `You previously extracted this puzzle structure from the image:

Rows: {rows}
Cols: {cols}

Shape:
{shape}

Regions:
{regions}

Constraints:
{constraints}

Looking at the original image again, verify if this extraction is accurate. If you see any errors or improvements needed, provide corrections in the same JSON format. If everything looks correct, respond with an empty corrections object.

Respond with ONLY valid JSON (no markdown):

{
  "is_correct": true,
  "corrections": {},
  "issues_found": []
}

OR if corrections needed:

{
  "is_correct": false,
  "corrections": {
    "rows": 5,
    "regions": "AAABC\\nDDDBC\\n...",
    "constraints": {
      "B": {"type": "sum", "op": "==", "value": 15}
    }
  },
  "issues_found": ["Region B boundaries were incorrect", "Constraint value was 15 not 12"]
}

Only include fields in "corrections" that need to be changed. Focus on:
1. Grid dimensions accuracy
2. Region boundary accuracy
3. Constraint value accuracy
4. Operator correctness (<, >, ==, !=)`;

async function verifyBoardExtraction(
  base64Image: string,
  apiKey: string,
  initialExtraction: BoardExtractionResult
): Promise<{
  success: boolean;
  corrections?: BoardExtractionResult;
  error?: string;
}> {
  try {
    const constraintsStr = JSON.stringify(initialExtraction.constraints, null, 2);
    const prompt = VERIFICATION_PROMPT.replace('{rows}', String(initialExtraction.rows))
      .replace('{cols}', String(initialExtraction.cols))
      .replace('{shape}', initialExtraction.shape.replace(/\\n/g, '\n'))
      .replace('{regions}', initialExtraction.regions.replace(/\\n/g, '\n'))
      .replace('{constraints}', constraintsStr);

    const mediaType = inferImageMediaTypeFromBase64(base64Image);
    const normalizedBase64 = normalizeBase64ImageData(base64Image);

    const response = await callClaudeWithFallback(apiKey, [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: { type: 'base64', media_type: mediaType, data: normalizedBase64 },
          },
          { type: 'text', text: prompt },
        ],
      },
    ]);

    const text = response.content.map(b => b.text).join('\n');
    const jsonMatch = text.match(/\{[\s\S]*\}/);

    if (!jsonMatch) {
      return { success: false, error: 'No JSON in verification response' };
    }

    const parsed = JSON.parse(jsonMatch[0]);

    if (parsed.is_correct) {
      return { success: true }; // No corrections needed
    }

    // Apply corrections to initial extraction
    const corrected: BoardExtractionResult = {
      ...initialExtraction,
      ...parsed.corrections,
    };

    return { success: true, corrections: corrected };
  } catch (error) {
    return {
      success: false,
      error: `Verification failed: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Pass 1: Board Extraction
// ════════════════════════════════════════════════════════════════════════════

/**
 * Optimized Board Extraction Prompt
 * Based on December 2025 research on prompt engineering for vision LLMs
 *
 * Key optimizations:
 * - Explicit step-by-step structure
 * - Clear JSON schema definition
 * - Negative examples to prevent common errors
 * - Conservative confidence scoring guidance
 */
const BOARD_EXTRACTION_PROMPT = `Analyze this NYT Pips puzzle screenshot. Extract the grid structure with maximum precision.

═══════════════════════════════════════════════════════════════════════════════
STEP 1: GRID DIMENSIONS (count carefully!)
═══════════════════════════════════════════════════════════════════════════════
- Count ROWS: Number of horizontal cell lines from TOP to BOTTOM
- Count COLS: Number of cells in the WIDEST row from LEFT to RIGHT
- Note: Corners or edges may have MISSING cells - include these in your count

═══════════════════════════════════════════════════════════════════════════════
STEP 2: GRID LOCATION (pixel coordinates)
═══════════════════════════════════════════════════════════════════════════════
Identify the EXACT pixel boundaries of the game grid:
- left: X-coordinate of the LEFTMOST cell edge
- top: Y-coordinate of the TOPMOST cell edge
- right: X-coordinate of the RIGHTMOST cell edge
- bottom: Y-coordinate of the BOTTOMMOST cell edge
- imageWidth: Full width of the image
- imageHeight: Full height of the image

═══════════════════════════════════════════════════════════════════════════════
STEP 3: SHAPE MAP (which cells exist)
═══════════════════════════════════════════════════════════════════════════════
For EACH cell position in the rows×cols grid:
- "." (dot) = cell EXISTS at this position
- "#" (hash) = NO cell here (hole/gap/missing)

IMPORTANT: Check ALL four corners for missing cells!
Join rows with \\n (backslash-n literal, not newline)

═══════════════════════════════════════════════════════════════════════════════
STEP 4: REGION MAP (colored areas)
═══════════════════════════════════════════════════════════════════════════════
For each cell that EXISTS, identify its colored region:
- Use letters A, B, C, D, E, F, G, H, I, J (in order of appearance)
- "#" for holes (same as shape)
- "." for cells with no visible color yet

Join rows with \\n (backslash-n literal)

═══════════════════════════════════════════════════════════════════════════════
STEP 5: CONSTRAINTS (read numbers/symbols)
═══════════════════════════════════════════════════════════════════════════════
Look for constraint text near each colored region:
- "12" or "=12" → {"type": "sum", "op": "==", "value": 12}
- "<10" → {"type": "sum", "op": "<", "value": 10}
- ">5" → {"type": "sum", "op": ">", "value": 5}
- "≠" or "all different" → {"type": "all_different"}
- "=" with no number → {"type": "all_equal"}

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only, NO markdown)
═══════════════════════════════════════════════════════════════════════════════
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
  "confidence": {
    "grid": 0.95,
    "regions": 0.88,
    "constraints": 0.92
  },
  "reasoning": "5x5 grid with holes at NE and SW corners..."
}

═══════════════════════════════════════════════════════════════════════════════
CONFIDENCE SCORING (be conservative!)
═══════════════════════════════════════════════════════════════════════════════
- 0.95-1.00: Absolutely certain
- 0.85-0.94: Very confident, minor uncertainty
- 0.70-0.84: Moderately confident, some ambiguity
- <0.70: Low confidence, significant uncertainty

Lower confidence if:
- Grid lines are unclear or partially visible
- Region colors are similar or hard to distinguish
- Constraint text is small, blurry, or partially obscured

═══════════════════════════════════════════════════════════════════════════════
CRITICAL ERRORS TO AVOID
═══════════════════════════════════════════════════════════════════════════════
❌ NEVER put region letters (A,B,C) in the "shape" field - only . and #
❌ NEVER use actual newlines - use \\n string literals
❌ NEVER add markdown code blocks (no \`\`\`)
❌ NEVER forget gridLocation pixel coordinates
❌ NEVER output anything except the JSON object

✅ ALWAYS use . for cells that exist in shape field
✅ ALWAYS use # for holes in BOTH shape and regions
✅ ALWAYS double-check grid dimensions by counting
✅ ALWAYS provide imageWidth and imageHeight

RESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT.`;

async function extractBoard(
  base64Image: string,
  apiKey: string
): Promise<{ success: boolean; data?: BoardExtractionResult; error?: string; reasoning?: string }> {
  try {
    const mediaType = inferImageMediaTypeFromBase64(base64Image);
    const normalizedBase64 = normalizeBase64ImageData(base64Image);

    console.log(`[DEBUG] extractBoard - Image media type detected: ${mediaType}`);
    console.log(`[DEBUG] extractBoard - Base64 length: ${normalizedBase64.length} chars`);
    console.log(
      `[DEBUG] extractBoard - Base64 first 50 chars: ${normalizedBase64.substring(0, 50)}...`
    );

    // Validate base64 format
    const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/;
    if (!base64Regex.test(normalizedBase64)) {
      const invalidChars = normalizedBase64.match(/[^A-Za-z0-9+/=]/g);
      return {
        success: false,
        error: `Invalid base64 characters detected: ${
          invalidChars?.slice(0, 10).join(', ') || 'unknown'
        }`,
      };
    }
    // Try to decode base64 to ensure it's valid (if atob is available)
    if (typeof atob !== 'undefined') {
      try {
        // Test decode a sample to ensure it's valid
        let testSample = normalizedBase64.substring(0, Math.min(100, normalizedBase64.length));
        // Add padding if needed
        while (testSample.length % 4 !== 0) {
          testSample += '=';
        }
        atob(testSample);
        console.log('[DEBUG] extractBoard - Base64 validation passed');
      } catch (decodeError) {
        console.error('[DEBUG] extractBoard - Base64 decode test failed:', decodeError);
        return {
          success: false,
          error: `Base64 data is not valid: ${
            decodeError instanceof Error ? decodeError.message : String(decodeError)
          }`,
        };
      }
    }
    const response = await callClaudeWithFallback(apiKey, [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: { type: 'base64', media_type: mediaType, data: normalizedBase64 },
          },
          { type: 'text', text: BOARD_EXTRACTION_PROMPT },
        ],
      },
    ]);

    const text = response.content.map(b => b.text).join('\n');

    // Try to parse JSON from the response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return { success: false, error: 'Could not parse board extraction response - no JSON found' };
    }

    // Fix multi-line strings in shape and regions fields
    // The AI sometimes returns strings split across lines like:
    // "shape": "...."
    //            "...."
    // We need to join these into a single string with \n separators
    let jsonString = jsonMatch[0];

    // Fix pattern: "field": "line1"
    //              "line2"
    //              "line3"
    // This matches a quoted string followed by indented quoted strings on subsequent lines
    const fixMultilineString = (fieldName: string, json: string): string => {
      // Pattern: "fieldName": "content"
      //          "more content" (indented)
      const pattern = new RegExp(`"${fieldName}"\\s*:\\s*"([^"]*)"\\s*\\n\\s*"([^"]*)"`, 'g');

      let fixed = json;
      let prevFixed = '';
      let iterations = 0;

      // Keep fixing until no more matches (handles multiple lines)
      while (fixed !== prevFixed && iterations < 10) {
        prevFixed = fixed;
        fixed = fixed.replace(pattern, (match, line1, line2) => {
          // Join with newline character
          return `"${fieldName}": "${line1}\\n${line2}"`;
        });
        iterations++;
      }

      return fixed;
    };

    jsonString = fixMultilineString('shape', jsonString);
    jsonString = fixMultilineString('regions', jsonString);

    let rawParsed;
    try {
      rawParsed = JSON.parse(jsonString);
    } catch (parseError) {
      // Fallback: try to fix by removing all newlines between quoted strings and joining them
      try {
        let fallbackFix = jsonMatch[0];
        // Remove newlines and extra quotes between string continuations
        fallbackFix = fallbackFix.replace(
          /("(?:shape|regions)"\s*:\s*")([^"]*)"\s*\n\s*"([^"]*)"\s*\n\s*"([^"]*)"\s*\n\s*"([^"]*)"/g,
          (match, prefix, l1, l2, l3, l4) => {
            return `${prefix}${l1}\\n${l2}\\n${l3}\\n${l4}"`;
          }
        );
        // Handle 3 lines
        fallbackFix = fallbackFix.replace(
          /("(?:shape|regions)"\s*:\s*")([^"]*)"\s*\n\s*"([^"]*)"\s*\n\s*"([^"]*)"/g,
          (match, prefix, l1, l2, l3) => {
            return `${prefix}${l1}\\n${l2}\\n${l3}"`;
          }
        );
        // Handle 2 lines
        fallbackFix = fallbackFix.replace(
          /("(?:shape|regions)"\s*:\s*")([^"]*)"\s*\n\s*"([^"]*)"/g,
          (match, prefix, l1, l2) => {
            return `${prefix}${l1}\\n${l2}"`;
          }
        );

        rawParsed = JSON.parse(fallbackFix);
      } catch (secondError) {
        return {
          success: false,
          error: `Invalid JSON in board response: ${parseError}. Fallback attempt: ${secondError}`,
        };
      }
    }

    // Validate with Zod schema
    const validationResult = BoardExtractionSchema.safeParse(rawParsed);
    if (!validationResult.success) {
      const errors = validationResult.error.issues.map(
        (e: z.ZodIssue) => `${e.path.join('.')}: ${e.message}`
      );
      return { success: false, error: `Board validation failed: ${errors.join(', ')}` };
    }

    const validated = validationResult.data;
    return {
      success: true,
      data: {
        rows: validated.rows,
        cols: validated.cols,
        shape: validated.shape,
        regions: validated.regions,
        constraints: validated.constraints || {},
        confidence: validated.confidence || { grid: 1.0, regions: 1.0, constraints: 1.0 },
      },
      reasoning: validated.reasoning || '',
    };
  } catch (error) {
    return {
      success: false,
      error: `Board extraction failed: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Pass 2: Domino Extraction
// ════════════════════════════════════════════════════════════════════════════

/**
 * Optimized Domino Extraction Prompt
 * Based on December 2025 research on pip/object counting with vision LLMs
 */
const DOMINO_EXTRACTION_PROMPT = `Extract the DOMINOES from this Pips puzzle screenshot.

═══════════════════════════════════════════════════════════════════════════════
CONTEXT (already extracted)
═══════════════════════════════════════════════════════════════════════════════
Grid: {rows}x{cols}
Shape: {shape}
Regions: {regions}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK: EXTRACT DOMINOES FROM THE TRAY
═══════════════════════════════════════════════════════════════════════════════
The dominoes are shown in a SEPARATE AREA (usually below or beside the main grid).
This is the "tray" or "bank" of available tiles to place.

DO NOT count cells on the grid - ONLY count dominoes in the tray area.

═══════════════════════════════════════════════════════════════════════════════
PIP COUNTING REFERENCE (memorize these patterns)
═══════════════════════════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────────┐
│ 0 pips │  BLANK - No dots at all (empty half)              │
│ 1 pip  │  ● - Single dot in CENTER                         │
│ 2 pips │  ●     - Two dots DIAGONAL (top-left, bottom-right) │
│        │    ●                                               │
│ 3 pips │  ●     - Three dots DIAGONAL LINE                  │
│        │   ●    (corner to corner)                          │
│        │    ●                                               │
│ 4 pips │  ● ●   - Four dots in CORNERS                      │
│        │  ● ●                                               │
│ 5 pips │  ● ●   - Four CORNERS + CENTER dot                 │
│        │   ●                                                │
│        │  ● ●                                               │
│ 6 pips │  ● ●   - Two COLUMNS of 3 dots each                │
│        │  ● ●   (6 dots total, arranged 3+3)                │
│        │  ● ●                                               │
└─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
EXTRACTION PROCEDURE
═══════════════════════════════════════════════════════════════════════════════
1. Locate the domino tray area (separate from the game grid)
2. Count total number of dominoes visible
3. For EACH domino, count pips on BOTH halves
4. Record as [left_pips, right_pips] or [top_pips, bottom_pips]
5. Scan systematically: left-to-right, top-to-bottom

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON only, NO markdown)
═══════════════════════════════════════════════════════════════════════════════
{
  "dominoes": [[6, 4], [5, 3], [2, 1], [0, 0], [4, 4], [3, 2]],
  "confidence": 0.92,
  "reasoning": "Found 6 dominoes in tray below grid. All pip counts clear."
}

═══════════════════════════════════════════════════════════════════════════════
CONFIDENCE SCORING
═══════════════════════════════════════════════════════════════════════════════
Set confidence < 0.80 if:
- Tray is partially cropped or cut off
- Any pips are too small to count clearly
- Image quality is poor
- Any dominoes overlap or are obscured

═══════════════════════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════════════════════
❌ DO NOT count cells ON THE GRID as dominoes
❌ DO NOT guess if pip count is unclear - lower confidence instead
❌ DO NOT add markdown formatting
❌ DO NOT output anything except the JSON object

✅ Count ONLY dominoes in the TRAY/BANK area
✅ Double-check each pip count before adding
✅ Each domino value must be 0-6 (integers only)
✅ Typical puzzle has 8-15 dominoes

RESPOND WITH ONLY THE JSON OBJECT. NO OTHER TEXT.`;

async function extractDominoes(
  base64Image: string,
  apiKey: string,
  board: BoardExtractionResult
): Promise<{
  success: boolean;
  data?: DominoExtractionResult;
  error?: string;
  reasoning?: string;
}> {
  const prompt = DOMINO_EXTRACTION_PROMPT.replace('{rows}', String(board.rows))
    .replace('{cols}', String(board.cols))
    .replace('{shape}', board.shape)
    .replace('{regions}', board.regions);

  try {
    const mediaType = inferImageMediaTypeFromBase64(base64Image);
    const normalizedBase64 = normalizeBase64ImageData(base64Image);
    const response = await callClaudeWithFallback(apiKey, [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: { type: 'base64', media_type: mediaType, data: normalizedBase64 },
          },
          { type: 'text', text: prompt },
        ],
      },
    ]);

    const text = response.content.map(b => b.text).join('\n');

    // Try to parse JSON from the response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return {
        success: false,
        error: 'Could not parse domino extraction response - no JSON found',
      };
    }

    let rawParsed;
    try {
      rawParsed = JSON.parse(jsonMatch[0]);
    } catch (parseError) {
      return { success: false, error: `Invalid JSON in domino response: ${parseError}` };
    }

    // Validate with Zod schema
    const validationResult = DominoExtractionSchema.safeParse(rawParsed);
    if (!validationResult.success) {
      const errors = validationResult.error.issues.map(
        (e: z.ZodIssue) => `${e.path.join('.')}: ${e.message}`
      );
      return { success: false, error: `Domino validation failed: ${errors.join(', ')}` };
    }

    const validated = validationResult.data;
    return {
      success: true,
      data: {
        dominoes: validated.dominoes as DominoPair[],
        confidence: validated.confidence || 1.0,
      },
      reasoning: validated.reasoning || '',
    };
  } catch (error) {
    return {
      success: false,
      error: `Domino extraction failed: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Claude API Call
// ════════════════════════════════════════════════════════════════════════════

type AnthropicErrorPayload = {
  type?: string;
  error?: { type?: string; message?: string };
  request_id?: string;
};

function tryParseAnthropicError(text: string): AnthropicErrorPayload | null {
  try {
    return JSON.parse(text) as AnthropicErrorPayload;
  } catch {
    return null;
  }
}

function isModelNotFoundResponse(status: number, bodyText: string): boolean {
  if (status !== 404) return false;
  const parsed = tryParseAnthropicError(bodyText);
  if (
    parsed?.error?.type === 'not_found_error' &&
    (parsed.error.message || '').startsWith('model:')
  ) {
    return true;
  }
  return bodyText.includes('not_found_error') && bodyText.includes('model:');
}

async function callClaudeWithFallback(
  apiKey: string,
  messages: ClaudeMessage[]
): Promise<ClaudeResponse> {
  let lastError: unknown = null;

  for (const model of MODEL_CANDIDATES) {
    try {
      return await callClaude(apiKey, messages, model);
    } catch (e) {
      lastError = e;
      const msg = e instanceof Error ? e.message : String(e);
      // Only fall back on "model not found". Anything else should surface immediately.
      if (msg.startsWith('API error: 404 - ')) {
        const body = msg.substring('API error: 404 - '.length);
        if (isModelNotFoundResponse(404, body)) {
          continue;
        }
      }
      if (msg.includes('not_found_error') && msg.includes('model:')) {
        continue;
      }
      throw e;
    }
  }

  throw new Error(
    `API error: none of the configured models were available (${MODEL_CANDIDATES.join(
      ', '
    )}). Last error: ${lastError instanceof Error ? lastError.message : String(lastError)}`
  );
}

async function callClaude(
  apiKey: string,
  messages: ClaudeMessage[],
  model: string
): Promise<ClaudeResponse> {
  // For Expo web, Anthropic requires this header to allow browser access.
  const extraHeaders: Record<string, string> =
    Platform.OS === 'web' ? { 'anthropic-dangerous-direct-browser-access': 'true' } : {};

  const requestBody: any = {
    model,
    max_tokens: 2048,
    messages,
  };

  // Try to use structured output mode if available (JSON mode)
  // This is supported in newer Claude models (claude-sonnet-4 and later)
  if (model.includes('claude-sonnet-4') || model.includes('claude-3-5-sonnet')) {
    // Note: As of API version 2023-06-01, structured output is not yet available
    // But we prepare for it here for future compatibility
    // requestBody.response_format = { type: 'json_object' };
  }

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      ...extraHeaders,
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API error: ${response.status} - ${error}`);
  }

  return await response.json();
}

// ════════════════════════════════════════════════════════════════════════════
// Convert AI Result to Builder State
// ════════════════════════════════════════════════════════════════════════════

/**
 * Convert AI extraction result to OverlayBuilder state updates
 */
export function convertAIResultToBuilderState(result: AIExtractionResult): {
  grid: Partial<GridState>;
  regions: Partial<RegionState>;
  constraints: Partial<ConstraintState>;
  dominoes: DominoPair[];
} {
  const { board, dominoes } = result;

  // Helper to normalize line breaks - AI may return \\n (escaped) or \n (actual newlines)
  const normalizeLineBreaks = (str: string): string => {
    // First replace literal backslash-n with newline, then normalize
    return str.replace(/\\n/g, '\n').trim();
  };

  // Parse shape to create holes array
  const normalizedShape = normalizeLineBreaks(board.shape);
  const shapeLines = normalizedShape
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0);
  const rows = shapeLines.length;
  const cols = Math.max(...shapeLines.map(line => line.length), 0);

  console.log('[DEBUG] convertAIResultToBuilderState: parsed shape', {
    originalShape: board.shape.substring(0, 100),
    normalizedShape: normalizedShape.substring(0, 100),
    rows,
    cols,
    shapeLines,
  });

  // Ensure each row has consistent column count (pad with '#' if needed)
  const holes: boolean[][] = shapeLines.map(line => {
    const chars = Array.from(line);
    // Pad short rows with '#' (holes)
    while (chars.length < cols) {
      chars.push('#');
    }
    return chars.map(char => char === '#');
  });

  // Parse regions
  const normalizedRegions = normalizeLineBreaks(board.regions);
  const regionLines = normalizedRegions
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0);

  const labelToIndex: Record<string, number> = {};
  let nextIndex = 0;

  const regionGrid: (number | null)[][] = regionLines.map((line, r) =>
    Array.from(line).map((char, c) => {
      if (char === '#' || holes[r]?.[c]) {
        return null;
      }
      if (!(char in labelToIndex)) {
        labelToIndex[char] = nextIndex++;
      }
      return labelToIndex[char];
    })
  );

  // Convert constraints
  const regionConstraints: Record<number, ConstraintDef> = {};
  for (const [label, constraint] of Object.entries(board.constraints)) {
    const index = labelToIndex[label];
    if (index !== undefined) {
      if (constraint.type === 'sum') {
        regionConstraints[index] = {
          type: 'sum',
          op: (constraint.op as '==' | '<' | '>' | '!=') || '==',
          value: constraint.value,
        };
      } else if (constraint.type === 'all_equal') {
        regionConstraints[index] = { type: 'all_equal' };
      } else if (constraint.type === 'all_different') {
        regionConstraints[index] = { type: 'all_different' };
      }
    }
  }

  // Calculate optimal bounds for the new grid dimensions
  // If gridLocation is provided, use it to calculate accurate bounds
  let bounds = DEFAULT_GRID_BOUNDS;

  if (board.gridLocation) {
    const loc = board.gridLocation;
    // Convert pixel coordinates to percentages
    bounds = {
      left: (loc.left / loc.imageWidth) * 100,
      top: (loc.top / loc.imageHeight) * 100,
      right: (loc.right / loc.imageWidth) * 100,
      bottom: (loc.bottom / loc.imageHeight) * 100,
    };

    console.log('[DEBUG] Calculated bounds from gridLocation:', {
      pixelCoords: loc,
      percentBounds: bounds,
    });
  } else {
    console.log('[DEBUG] No gridLocation provided, using default bounds');
  }

  return {
    grid: { rows, cols, holes, bounds },
    regions: { regionGrid, palette: { ...DEFAULT_PALETTE, selectedIndex: 0 } },
    constraints: { regionConstraints, selectedRegion: null },
    dominoes: dominoes.dominoes,
  };
}

/**
 * AI Extraction Service
 * Uses Claude's vision capabilities to extract puzzle data from screenshots
 */

import { Platform } from 'react-native';
import { z } from 'zod';
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

// Prefer newest supported models, but gracefully fall back if the account
// doesn't have access to a given model ID.
const MODEL_CANDIDATES = [
  'claude-sonnet-4-20250514',
  // Older fallbacks (may be retired for some accounts, but cheap to try)
  'claude-3-5-sonnet-20240620',
  'claude-3-opus-20240229',
] as const;

// ════════════════════════════════════════════════════════════════════════════
// Zod Schemas for AI Response Validation
// ════════════════════════════════════════════════════════════════════════════

const ConstraintSchema = z.object({
  type: z.enum(['sum', 'all_equal', 'all_different']),
  op: z.enum(['==', '<', '>', '!=']).optional(),
  value: z.number().optional(),
});

const BoardExtractionSchema = z.object({
  rows: z.number().min(2).max(12),
  cols: z.number().min(2).max(12),
  shape: z.string().min(1),
  regions: z.string().min(1),
  constraints: z.record(z.string(), ConstraintSchema).optional(),
  reasoning: z.string().optional(),
});

const DominoExtractionSchema = z.object({
  dominoes: z.array(z.tuple([z.number().min(0).max(6), z.number().min(0).max(6)])),
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

function normalizeBase64ImageData(input: string): string {
  // #region agent log
  const logEntry = {
    location: 'aiExtraction.ts:76',
    message: 'normalizeBase64ImageData entry',
    data: {
      inputLength: input?.length || 0,
      inputPrefix: input?.substring(0, 50) || 'null',
      hasDataPrefix: input?.startsWith('data:') || false,
    },
    timestamp: Date.now(),
    sessionId: 'debug-session',
    runId: 'run1',
    hypothesisId: 'A',
  };
  console.log('[DEBUG] normalizeBase64ImageData entry:', logEntry);
  fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(logEntry),
  }).catch(() => {});
  // #endregion
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

  // #region agent log
  const logExit = {
    location: 'aiExtraction.ts:90',
    message: 'normalizeBase64ImageData exit',
    data: {
      outputLength: s?.length || 0,
      outputPrefix: s?.substring(0, 50) || 'null',
      isEmpty: s?.length === 0,
    },
    timestamp: Date.now(),
    sessionId: 'debug-session',
    runId: 'run1',
    hypothesisId: 'A',
  };
  console.log('[DEBUG] normalizeBase64ImageData exit:', logExit);
  fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(logExit),
  }).catch(() => {});
  // #endregion
  return s;
}

function inferImageMediaTypeFromBase64(base64: string): string {
  // #region agent log
  const logEntry = {
    location: 'aiExtraction.ts:93',
    message: 'inferImageMediaTypeFromBase64 entry',
    data: { base64Length: base64?.length || 0, base64Prefix: base64?.substring(0, 20) || 'null' },
    timestamp: Date.now(),
    sessionId: 'debug-session',
    runId: 'run1',
    hypothesisId: 'B',
  };
  console.log('[DEBUG] inferImageMediaTypeFromBase64 entry:', logEntry);
  fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(logEntry),
  }).catch(() => {});
  // #endregion
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

  // #region agent log
  const logExit = {
    location: `aiExtraction.ts:${detectedType === 'unknown' ? '106' : '97'}`,
    message: detectedType === 'unknown' ? 'defaulting to JPEG' : `detected ${detectedType}`,
    data: { mediaType, detectedType, base64Prefix: b64?.substring(0, 20) || 'null' },
    timestamp: Date.now(),
    sessionId: 'debug-session',
    runId: 'run1',
    hypothesisId: 'B',
  };
  console.log('[DEBUG] Media type detection result:', logExit);
  fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(logExit),
  }).catch(() => {});
  // #endregion
  return mediaType;
}

// ════════════════════════════════════════════════════════════════════════════
// Main Extraction Function
// ════════════════════════════════════════════════════════════════════════════

/**
 * Extract puzzle data from a screenshot using Claude's vision
 * Uses a two-pass approach: first extracts board, then extracts dominoes
 * Supports partial success - board results are returned even if dominoes fail
 */
export async function extractPuzzleFromImage(
  base64Image: string,
  apiKey: string,
  onProgress?: (progress: ExtractionProgress) => void
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
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      location: 'aiExtraction.ts:133',
      message: 'normalized image in extractPuzzleFromImage',
      data: {
        originalLength: base64Image?.length || 0,
        normalizedLength: normalizedImage?.length || 0,
        isEmpty: !normalizedImage || normalizedImage.length === 0,
      },
      timestamp: Date.now(),
      sessionId: 'debug-session',
      runId: 'run1',
      hypothesisId: 'E',
    }),
  }).catch(() => {});
  // #endregion
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

    // Pass 2: Extract dominoes (independent - partial success allowed)
    console.log('[DEBUG] extractPuzzleFromImage - Starting domino extraction...');
    onProgress?.({ step: 'dominoes', message: 'Extracting dominoes...' });
    const dominoExtractStartTime = Date.now();
    const dominoResult = await extractDominoes(normalizedImage, trimmedKey, boardResult.data);
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
          board: boardResult.data,
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
        board: boardResult.data,
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
// Pass 1: Board Extraction
// ════════════════════════════════════════════════════════════════════════════

const BOARD_EXTRACTION_PROMPT = `You are analyzing a screenshot of a Pips puzzle from the NYT Games app.

Your task is to extract the board structure. Look carefully at the image and identify:

1. **Grid dimensions**: Count the rows and columns of cells
2. **Board shape**: Which cells exist vs which are empty/holes (cells that are blacked out or missing)
3. **Regions**: The colored areas that group cells together. Each region has a unique color.
4. **Constraints**: The numbers or symbols shown for each region indicating sum targets or rules

Respond with a JSON object (no markdown, just raw JSON):
{
  "rows": <number>,
  "cols": <number>,
  "shape": "<ASCII art using . for cells and # for holes, with \\n between rows>",
  "regions": "<ASCII art using A-J for regions matching shape dimensions, with \\n between rows>",
  "constraints": {
    "A": {"type": "sum", "op": "==", "value": <number>},
    "B": {"type": "sum", "op": "<", "value": <number>},
    "C": {"type": "all_equal"}
  },
  "reasoning": "<brief explanation of what you see>"
}

IMPORTANT: The "shape" and "regions" fields must be single JSON strings with \\n (backslash-n) characters to separate rows. Do NOT split them across multiple lines in the JSON. For example:
  "shape": "....\\n....\\n....\\n...."
NOT:
  "shape": "...."
           "...."

Notes:
- Use A, B, C... for region labels based on position (top-left first, then left-to-right, top-to-bottom)
- Constraint types: "sum" with op "==" or "<" or ">", or "all_equal" for matching pip values
- The shape and regions strings must have the same dimensions (same number of rows and columns)
- If a region has no constraint shown, omit it from the constraints object`;

async function extractBoard(
  base64Image: string,
  apiKey: string
): Promise<{ success: boolean; data?: BoardExtractionResult; error?: string; reasoning?: string }> {
  try {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        location: 'aiExtraction.ts:220',
        message: 'extractBoard entry',
        data: {
          base64Length: base64Image?.length || 0,
          base64Prefix: base64Image?.substring(0, 30) || 'null',
          base64Suffix: base64Image?.substring(Math.max(0, base64Image?.length - 30)) || 'null',
          isEmpty: base64Image?.length === 0,
        },
        timestamp: Date.now(),
        sessionId: 'debug-session',
        runId: 'run1',
        hypothesisId: 'C',
      }),
    }).catch(() => {});
    // #endregion
    const mediaType = inferImageMediaTypeFromBase64(base64Image);
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        location: 'aiExtraction.ts:225',
        message: 'before API call',
        data: {
          mediaType,
          base64Length: base64Image?.length || 0,
          normalizedLength: normalizeBase64ImageData(base64Image)?.length || 0,
        },
        timestamp: Date.now(),
        sessionId: 'debug-session',
        runId: 'run1',
        hypothesisId: 'C',
      }),
    }).catch(() => {});
    // #endregion
    const normalizedBase64 = normalizeBase64ImageData(base64Image);
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
      } catch (decodeError) {
        return {
          success: false,
          error: `Base64 data is not valid: ${
            decodeError instanceof Error ? decodeError.message : String(decodeError)
          }`,
        };
      }
    }
    // #region agent log
    const logData = {
      location: 'aiExtraction.ts:228',
      message: 'using normalized base64',
      data: {
        normalizedLength: normalizedBase64?.length || 0,
        normalizedPrefix: normalizedBase64?.substring(0, 30) || 'null',
        normalizedSuffix:
          normalizedBase64?.substring(Math.max(0, normalizedBase64.length - 30)) || 'null',
        mediaType,
        isValidBase64: base64Regex.test(normalizedBase64),
        payloadSize: JSON.stringify([
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
        ]).length,
      },
      timestamp: Date.now(),
      sessionId: 'debug-session',
      runId: 'run1',
      hypothesisId: 'C',
    };
    console.log('[DEBUG] extractBoard - Before API call:', JSON.stringify(logData, null, 2));
    console.log('[DEBUG] extractBoard - Base64 first 50 chars:', normalizedBase64.substring(0, 50));
    console.log(
      '[DEBUG] extractBoard - Base64 last 50 chars:',
      normalizedBase64.substring(Math.max(0, normalizedBase64.length - 50))
    );
    console.log('[DEBUG] extractBoard - Media type:', mediaType);
    fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(logData),
    }).catch(() => {});
    // #endregion
    console.log('[DEBUG] extractBoard - About to call API...');
    const apiCallStartTime = Date.now();
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
    const apiCallDuration = Date.now() - apiCallStartTime;
    console.log(`[DEBUG] extractBoard - API call completed in ${apiCallDuration}ms`);
    console.log('[DEBUG] extractBoard - Response received:', {
      contentLength: response.content?.length || 0,
      stopReason: response.stop_reason,
      firstContentText: response.content?.[0]?.text?.substring(0, 200) || 'none',
    });

    const text = response.content.map(b => b.text).join('\n');
    console.log('[DEBUG] extractBoard - Extracted text length:', text.length);
    console.log(
      '[DEBUG] extractBoard - Full response text (first 1000 chars):',
      text.substring(0, 1000)
    );
    console.log(
      '[DEBUG] extractBoard - Full response text (last 500 chars):',
      text.substring(Math.max(0, text.length - 500))
    );

    // Try to parse JSON from the response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      console.error('[DEBUG] extractBoard - No JSON found in response. Full text:', text);
      return { success: false, error: 'Could not parse board extraction response - no JSON found' };
    }

    console.log('[DEBUG] extractBoard - Found JSON match, length:', jsonMatch[0].length);

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

    console.log(
      '[DEBUG] extractBoard - Fixed JSON string (first 500 chars):',
      jsonString.substring(0, 500)
    );

    let rawParsed;
    try {
      rawParsed = JSON.parse(jsonString);
      console.log(
        '[DEBUG] extractBoard - Parsed JSON successfully:',
        JSON.stringify(rawParsed, null, 2)
      );
    } catch (parseError) {
      console.error('[DEBUG] extractBoard - JSON parse error:', parseError);
      console.error(
        '[DEBUG] extractBoard - JSON string that failed (first 1000 chars):',
        jsonString.substring(0, 1000)
      );

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
        console.log('[DEBUG] extractBoard - Parsed with fallback fix');
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
      console.error('[DEBUG] extractBoard - Validation failed:', errors);
      console.error('[DEBUG] extractBoard - Raw parsed data:', JSON.stringify(rawParsed, null, 2));
      return { success: false, error: `Board validation failed: ${errors.join(', ')}` };
    }

    console.log('[DEBUG] extractBoard - Validation passed');

    const validated = validationResult.data;
    console.log('[DEBUG] extractBoard - Final validated data:', {
      rows: validated.rows,
      cols: validated.cols,
      shapeLength: validated.shape?.length,
      shapePreview: validated.shape?.substring(0, 200),
      regionsLength: validated.regions?.length,
      regionsPreview: validated.regions?.substring(0, 200),
      constraintsCount: Object.keys(validated.constraints || {}).length,
      constraints: validated.constraints,
      reasoningLength: validated.reasoning?.length || 0,
    });
    return {
      success: true,
      data: {
        rows: validated.rows,
        cols: validated.cols,
        shape: validated.shape,
        regions: validated.regions,
        constraints: validated.constraints || {},
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

const DOMINO_EXTRACTION_PROMPT = `You are analyzing a screenshot of a Pips puzzle from the NYT Games app.

The board structure has already been identified as:
- Grid: {rows}x{cols}
- Shape:
{shape}

- Regions:
{regions}

Your task is to extract the DOMINOES shown in the puzzle. Each domino is a pair of connected cells with pip values (0-6 dots).

Look at the REFERENCE DOMINOES shown in the puzzle (usually displayed below or beside the main grid). These are the available dominoes that need to be placed.

Respond with a JSON object (no markdown, just raw JSON):
{
  "dominoes": [[pip1, pip2], [pip1, pip2], ...],
  "reasoning": "<brief explanation of the dominoes you identified>"
}

Notes:
- Each domino is represented as [pip1, pip2] where pip values are 0-6
- List all dominoes shown in the reference/available dominoes area
- Order doesn't matter
- Common dominoes: [6,1] means one half has 6 pips, other half has 1 pip`;

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
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        location: 'aiExtraction.ts:325',
        message: 'extractDominoes before API call',
        data: { base64Length: base64Image?.length || 0 },
        timestamp: Date.now(),
        sessionId: 'debug-session',
        runId: 'run1',
        hypothesisId: 'C',
      }),
    }).catch(() => {});
    // #endregion
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
      data: { dominoes: validated.dominoes as DominoPair[] },
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
  // #region agent log
  const imageContent = messages[0]?.content?.find(
    (c): c is { type: 'image'; source: { type: 'base64'; media_type: string; data: string } } =>
      c.type === 'image'
  );
  const logEntry = {
    location: 'aiExtraction.ts:441',
    message: 'callClaude entry',
    data: {
      model,
      apiKeyLength: apiKey?.length || 0,
      hasImageContent: !!imageContent,
      imageMediaType: imageContent?.source?.media_type || 'none',
      imageDataLength: imageContent?.source?.data?.length || 0,
      imageDataPrefix: imageContent?.source?.data?.substring(0, 30) || 'null',
      imageDataSuffix:
        imageContent?.source?.data?.substring(
          Math.max(0, (imageContent?.source?.data?.length || 0) - 30)
        ) || 'null',
      payloadSize: JSON.stringify({ model, max_tokens: 2048, messages }).length,
    },
    timestamp: Date.now(),
    sessionId: 'debug-session',
    runId: 'run1',
    hypothesisId: 'D',
  };
  console.log('[DEBUG] callClaude - Request details:', JSON.stringify(logEntry, null, 2));
  if (imageContent) {
    console.log('[DEBUG] callClaude - Image content structure:', {
      type: imageContent.type,
      sourceType: imageContent.source.type,
      mediaType: imageContent.source.media_type,
      dataLength: imageContent.source.data?.length,
      dataFirst50: imageContent.source.data?.substring(0, 50),
      dataLast50: imageContent.source.data?.substring(
        Math.max(0, (imageContent.source.data?.length || 0) - 50)
      ),
    });
  }
  fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(logEntry),
  }).catch(() => {});
  // #endregion
  // For Expo web, Anthropic requires this header to allow browser access.
  const extraHeaders: Record<string, string> =
    Platform.OS === 'web' ? { 'anthropic-dangerous-direct-browser-access': 'true' } : {};

  const requestBody = {
    model,
    max_tokens: 2048,
    messages,
  };

  console.log('[DEBUG] callClaude - Request body (without image data):', {
    model: requestBody.model,
    max_tokens: requestBody.max_tokens,
    messagesCount: requestBody.messages.length,
    firstMessageRole: requestBody.messages[0]?.role,
    firstMessageContentTypes: requestBody.messages[0]?.content?.map((c: any) => c.type),
  });
  console.log('[DEBUG] callClaude - About to make fetch request...');
  const fetchStartTime = Date.now();

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

  // #region agent log
  const fetchDuration = Date.now() - fetchStartTime;
  console.log(`[DEBUG] callClaude - Fetch completed in ${fetchDuration}ms`);
  const responseLog = {
    location: 'aiExtraction.ts:465',
    message: 'API response received',
    data: {
      status: response.status,
      statusText: response.statusText,
      ok: response.ok,
      fetchDuration,
    },
    timestamp: Date.now(),
    sessionId: 'debug-session',
    runId: 'run1',
    hypothesisId: 'D',
  };
  console.log('[DEBUG] callClaude - API response received:', responseLog);
  fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(responseLog),
  }).catch(() => {});
  // #endregion

  if (!response.ok) {
    const error = await response.text();
    // #region agent log
    const logData = {
      location: 'aiExtraction.ts:467',
      message: 'API error response',
      data: {
        status: response.status,
        errorText: error,
        errorLength: error?.length || 0,
        errorPrefix: error?.substring(0, 200) || 'null',
        fullError: error, // Include full error for debugging
      },
      timestamp: Date.now(),
      sessionId: 'debug-session',
      runId: 'run1',
      hypothesisId: 'D',
    };
    console.error('[DEBUG] API error response:', JSON.stringify(logData, null, 2));
    console.error('[DEBUG] Full error text:', error);
    try {
      const errorJson = JSON.parse(error);
      console.error('[DEBUG] Parsed error JSON:', JSON.stringify(errorJson, null, 2));
    } catch (e) {
      console.error('[DEBUG] Error text is not JSON:', e);
    }
    fetch('http://127.0.0.1:7242/ingest/9150e6c0-f2ac-4ac3-b8cb-698ec1abb2d7', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(logData),
    }).catch(() => {});
    // #endregion
    throw new Error(`API error: ${response.status} - ${error}`);
  }

  console.log('[DEBUG] callClaude - Parsing JSON response...');
  const jsonStartTime = Date.now();
  const result = await response.json();
  const jsonDuration = Date.now() - jsonStartTime;
  console.log(`[DEBUG] callClaude - JSON parsed in ${jsonDuration}ms`);
  return result;
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
  console.log('[DEBUG] convertAIResultToBuilderState - Input result:', {
    boardRows: result.board.rows,
    boardCols: result.board.cols,
    shapeLength: result.board.shape?.length,
    shapePreview: result.board.shape?.substring(0, 200),
    regionsLength: result.board.regions?.length,
    regionsPreview: result.board.regions?.substring(0, 200),
    constraintsCount: Object.keys(result.board.constraints || {}).length,
    dominoesCount: result.dominoes.dominoes?.length || 0,
  });
  const { board, dominoes } = result;

  // Parse shape to create holes array
  const shapeLines = board.shape
    .trim()
    .split('\n')
    .map(line => line.trim());
  const rows = shapeLines.length;
  const cols = shapeLines[0]?.length || 0;

  const holes: boolean[][] = shapeLines.map(line => Array.from(line).map(char => char === '#'));

  console.log('[DEBUG] convertAIResultToBuilderState - Parsed shape:', {
    shapeLinesCount: shapeLines.length,
    shapeLines: shapeLines,
    calculatedRows: rows,
    calculatedCols: cols,
    holesCount: holes.flat().filter(h => h).length,
  });

  // Parse regions
  const regionLines = board.regions
    .trim()
    .split('\n')
    .map(line => line.trim());

  console.log('[DEBUG] convertAIResultToBuilderState - Parsed regions:', {
    regionLinesCount: regionLines.length,
    regionLines: regionLines,
  });

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

  console.log('[DEBUG] convertAIResultToBuilderState - Region mapping:', {
    labelToIndex,
    regionGridDimensions: `${regionGrid.length}x${regionGrid[0]?.length || 0}`,
    regionGridPreview: regionGrid.slice(0, 2).map(row => row.slice(0, 4)),
  });

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
  // Use default bounds if image dimensions aren't available, otherwise calculate optimal
  let bounds = DEFAULT_GRID_BOUNDS;
  // Note: We can't calculate optimal bounds here without image dimensions,
  // so we'll use default bounds. The user can adjust in Step 1 if needed.
  // Alternatively, we could pass image dimensions to this function.

  const convertedState = {
    grid: { rows, cols, holes, bounds },
    regions: { regionGrid, palette: { ...DEFAULT_PALETTE, selectedIndex: 0 } },
    constraints: { regionConstraints, selectedRegion: null },
    dominoes: dominoes.dominoes,
  };

  console.log('[DEBUG] convertAIResultToBuilderState - Final result:', {
    gridRows: convertedState.grid.rows,
    gridCols: convertedState.grid.cols,
    gridBounds: convertedState.grid.bounds,
    holesArray: convertedState.grid.holes?.map(row => row.map(h => (h ? '#' : '.')).join('')),
    regionGridRows: convertedState.regions.regionGrid?.length,
    regionGridCols: convertedState.regions.regionGrid?.[0]?.length,
    constraintsCount: Object.keys(convertedState.constraints.regionConstraints || {}).length,
    dominoesCount: convertedState.dominoes.length,
  });

  return convertedState;
}

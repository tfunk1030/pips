/**
 * AI Extraction Service
 * Uses Claude's vision capabilities to extract puzzle data from screenshots
 */

import { z } from 'zod';
import {
  AIExtractionResult,
  BoardExtractionResult,
  DominoExtractionResult,
  DominoPair,
  GridState,
  RegionState,
  ConstraintState,
  ConstraintDef,
  DEFAULT_PALETTE,
} from '../model/overlayTypes';

// ════════════════════════════════════════════════════════════════════════════
// Zod Schemas for AI Response Validation
// ════════════════════════════════════════════════════════════════════════════

const ConstraintSchema = z.object({
  type: z.enum(['sum', 'all_equal']),
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
  dominoes: z.array(
    z.tuple([
      z.number().min(0).max(6),
      z.number().min(0).max(6),
    ])
  ),
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
  if (!apiKey) {
    return { success: false, error: 'API key is required' };
  }

  try {
    // Pass 1: Extract board structure
    onProgress?.({ step: 'board', message: 'Analyzing board structure...' });
    const boardResult = await extractBoard(base64Image, apiKey);

    if (!boardResult.success || !boardResult.data) {
      return { success: false, error: boardResult.error || 'Failed to extract board' };
    }

    // Pass 2: Extract dominoes (independent - partial success allowed)
    onProgress?.({ step: 'dominoes', message: 'Extracting dominoes...' });
    const dominoResult = await extractDominoes(base64Image, apiKey, boardResult.data);

    // If dominoes fail, return partial success with board data
    if (!dominoResult.success || !dominoResult.data) {
      onProgress?.({ step: 'done', message: 'Extraction partial - dominoes failed' });
      return {
        success: true,
        partial: true,
        result: {
          board: boardResult.data,
          dominoes: { dominoes: [] }, // Empty dominoes - user can add manually
          reasoning: `Board: ${boardResult.reasoning}\n\nDominoes: FAILED - ${dominoResult.error || 'Unknown error'}. Please add dominoes manually.`,
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
  "shape": "<ASCII art using . for cells and # for holes, one row per line>",
  "regions": "<ASCII art using A-J for regions matching shape dimensions>",
  "constraints": {
    "A": {"type": "sum", "op": "==", "value": <number>},
    "B": {"type": "sum", "op": "<", "value": <number>},
    "C": {"type": "all_equal"}
  },
  "reasoning": "<brief explanation of what you see>"
}

Notes:
- Use A, B, C... for region labels based on position (top-left first, then left-to-right, top-to-bottom)
- Constraint types: "sum" with op "==" or "<" or ">", or "all_equal" for matching pip values
- The shape and regions strings must have the same dimensions
- If a region has no constraint shown, omit it from the constraints object`;

async function extractBoard(
  base64Image: string,
  apiKey: string
): Promise<{ success: boolean; data?: BoardExtractionResult; error?: string; reasoning?: string }> {
  try {
    const response = await callClaude(apiKey, [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: base64Image } },
          { type: 'text', text: BOARD_EXTRACTION_PROMPT },
        ],
      },
    ]);

    const text = response.content[0].text;

    // Try to parse JSON from the response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return { success: false, error: 'Could not parse board extraction response - no JSON found' };
    }

    let rawParsed;
    try {
      rawParsed = JSON.parse(jsonMatch[0]);
    } catch (parseError) {
      return { success: false, error: `Invalid JSON in board response: ${parseError}` };
    }

    // Validate with Zod schema
    const validationResult = BoardExtractionSchema.safeParse(rawParsed);
    if (!validationResult.success) {
      const errors = validationResult.error.issues.map((e: z.ZodIssue) => `${e.path.join('.')}: ${e.message}`);
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
): Promise<{ success: boolean; data?: DominoExtractionResult; error?: string; reasoning?: string }> {
  const prompt = DOMINO_EXTRACTION_PROMPT
    .replace('{rows}', String(board.rows))
    .replace('{cols}', String(board.cols))
    .replace('{shape}', board.shape)
    .replace('{regions}', board.regions);

  try {
    const response = await callClaude(apiKey, [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: base64Image } },
          { type: 'text', text: prompt },
        ],
      },
    ]);

    const text = response.content[0].text;

    // Try to parse JSON from the response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return { success: false, error: 'Could not parse domino extraction response - no JSON found' };
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
      const errors = validationResult.error.issues.map((e: z.ZodIssue) => `${e.path.join('.')}: ${e.message}`);
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

async function callClaude(apiKey: string, messages: ClaudeMessage[]): Promise<ClaudeResponse> {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 2048,
      messages,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API error: ${response.status} - ${error}`);
  }

  return response.json();
}

// ════════════════════════════════════════════════════════════════════════════
// Convert AI Result to Builder State
// ════════════════════════════════════════════════════════════════════════════

/**
 * Convert AI extraction result to OverlayBuilder state updates
 */
export function convertAIResultToBuilderState(
  result: AIExtractionResult
): {
  grid: Partial<GridState>;
  regions: Partial<RegionState>;
  constraints: Partial<ConstraintState>;
  dominoes: DominoPair[];
} {
  const { board, dominoes } = result;

  // Parse shape to create holes array
  const shapeLines = board.shape.trim().split('\n').map(line => line.trim());
  const rows = shapeLines.length;
  const cols = shapeLines[0]?.length || 0;

  const holes: boolean[][] = shapeLines.map(line =>
    Array.from(line).map(char => char === '#')
  );

  // Parse regions
  const regionLines = board.regions.trim().split('\n').map(line => line.trim());
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
      }
    }
  }

  return {
    grid: { rows, cols, holes },
    regions: { regionGrid, palette: { ...DEFAULT_PALETTE, selectedIndex: 0 } },
    constraints: { regionConstraints, selectedRegion: null },
    dominoes: dominoes.dominoes,
  };
}

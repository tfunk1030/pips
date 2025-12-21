/**
 * CV Extraction Client
 * Calls Python CV service for pixel-accurate grid geometry detection
 *
 * The CV service handles:
 * - Grid line detection
 * - Cell boundary detection
 * - Hole detection (critical for accuracy)
 * - Grid dimension inference
 *
 * This is then fed to AI for semantic interpretation (regions, constraints)
 */

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

export interface CellBounds {
  x: number;
  y: number;
  width: number;
  height: number;
  row: number;
  col: number;
}

export interface CVExtractionResult {
  success: boolean;
  error?: string;

  // Grid structure
  rows: number;
  cols: number;

  // Cell data
  cells: CellBounds[];

  // Shape string (. = cell, # = hole)
  shape: string;

  // Grid bounds in image coordinates
  gridBounds?: {
    left: number;
    top: number;
    right: number;
    bottom: number;
    imageWidth: number;
    imageHeight: number;
  };

  // Timing
  extractionMs: number;
}

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

// Default CV service URL - can be overridden via settings
let CV_SERVICE_URL = 'http://localhost:8080';

/**
 * Configure the CV service URL
 */
export function setCVServiceURL(url: string): void {
  CV_SERVICE_URL = url;
  console.log(`[CV] Service URL set to: ${url}`);
}

/**
 * Get the current CV service URL
 */
export function getCVServiceURL(): string {
  return CV_SERVICE_URL;
}

// ════════════════════════════════════════════════════════════════════════════
// API Client
// ════════════════════════════════════════════════════════════════════════════

/**
 * Check if CV service is available
 */
export async function checkCVServiceHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${CV_SERVICE_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Extract grid geometry using CV service
 */
export async function extractGeometryWithCV(
  base64Image: string,
  options: { lowerHalfOnly?: boolean } = {}
): Promise<CVExtractionResult> {
  const { lowerHalfOnly = true } = options;

  try {
    console.log('[CV] Calling CV service for geometry extraction...');
    const startTime = Date.now();

    const response = await fetch(`${CV_SERVICE_URL}/extract-geometry`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64Image,
        lower_half_only: lowerHalfOnly,
      }),
    });

    if (!response.ok) {
      throw new Error(`CV service error: ${response.status}`);
    }

    const data = await response.json();
    const totalMs = Date.now() - startTime;

    console.log(
      `[CV] Extraction complete in ${totalMs}ms - ${data.rows}x${data.cols} grid, ${
        data.cells?.length || 0
      } cells`
    );

    // Transform response to camelCase
    return {
      success: data.success,
      error: data.error,
      rows: data.rows || 0,
      cols: data.cols || 0,
      cells: (data.cells || []).map((c: any) => ({
        x: c.x,
        y: c.y,
        width: c.width,
        height: c.height,
        row: c.row,
        col: c.col,
      })),
      shape: data.shape || '',
      gridBounds: data.grid_bounds
        ? {
            left: data.grid_bounds.left,
            top: data.grid_bounds.top,
            right: data.grid_bounds.right,
            bottom: data.grid_bounds.bottom,
            imageWidth: data.grid_bounds.imageWidth,
            imageHeight: data.grid_bounds.imageHeight,
          }
        : undefined,
      extractionMs: data.extraction_ms || totalMs,
    };
  } catch (error) {
    console.error('[CV] Extraction failed:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'CV extraction failed',
      rows: 0,
      cols: 0,
      cells: [],
      shape: '',
      extractionMs: 0,
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Puzzle Region Cropping (for Hybrid Extraction)
// ════════════════════════════════════════════════════════════════════════════

export interface CropResult {
  success: boolean;
  error?: string;
  croppedImage?: string; // base64 PNG of just the puzzle
  bounds?: {
    x: number;
    y: number;
    width: number;
    height: number;
    originalWidth: number;
    originalHeight: number;
  };
  // Actual grid bounds (without padding) - use for overlay alignment
  gridBounds?: {
    x: number;
    y: number;
    width: number;
    height: number;
    originalWidth: number;
    originalHeight: number;
  };
  extractionMs: number;
}

/**
 * Crop image to domino tray region only (below the puzzle grid)
 * Returns a focused image for better domino extraction accuracy
 */
export async function cropDominoRegion(
  base64Image: string,
  puzzleBottomY?: number
): Promise<CropResult> {
  try {
    console.log('[CV] Cropping domino region...');
    const startTime = Date.now();

    const response = await fetch(`${CV_SERVICE_URL}/crop-dominoes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64Image,
        puzzle_bottom_y: puzzleBottomY,
      }),
    });

    if (!response.ok) {
      throw new Error(`CV service error: ${response.status}`);
    }

    const data = await response.json();
    const totalMs = Date.now() - startTime;

    if (data.success) {
      console.log(
        `[CV] Domino crop in ${totalMs}ms - ${data.bounds?.width}x${data.bounds?.height}`
      );
    }

    return {
      success: data.success,
      error: data.error,
      croppedImage: data.cropped_image,
      bounds: data.bounds
        ? {
            x: data.bounds.x,
            y: data.bounds.y,
            width: data.bounds.width,
            height: data.bounds.height,
            originalWidth: data.bounds.original_width,
            originalHeight: data.bounds.original_height,
          }
        : undefined,
      extractionMs: data.extraction_ms || totalMs,
    };
  } catch (error) {
    console.error('[CV] Domino crop failed:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Domino crop failed',
      extractionMs: 0,
    };
  }
}

/**
 * Crop image to puzzle region only (excludes dominoes, UI elements)
 * Returns a smaller, cleaner image for AI analysis
 */
export async function cropPuzzleRegion(base64Image: string): Promise<CropResult> {
  try {
    console.log('[CV] Cropping puzzle region...');
    const startTime = Date.now();

    const response = await fetch(`${CV_SERVICE_URL}/crop-puzzle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image }),
    });

    if (!response.ok) {
      throw new Error(`CV service error: ${response.status}`);
    }

    const data = await response.json();
    const totalMs = Date.now() - startTime;

    if (data.success) {
      console.log(`[CV] Cropped in ${totalMs}ms - ${data.bounds?.width}x${data.bounds?.height}`);
    }

    return {
      success: data.success,
      error: data.error,
      croppedImage: data.cropped_image,
      bounds: data.bounds
        ? {
            x: data.bounds.x,
            y: data.bounds.y,
            width: data.bounds.width,
            height: data.bounds.height,
            originalWidth: data.bounds.original_width,
            originalHeight: data.bounds.original_height,
          }
        : undefined,
      // Actual grid bounds (without padding) for overlay alignment
      gridBounds: data.grid_bounds
        ? {
            x: data.grid_bounds.x,
            y: data.grid_bounds.y,
            width: data.grid_bounds.width,
            height: data.grid_bounds.height,
            originalWidth: data.grid_bounds.original_width,
            originalHeight: data.grid_bounds.original_height,
          }
        : undefined,
      extractionMs: data.extraction_ms || totalMs,
    };
  } catch (error) {
    console.error('[CV] Crop failed:', error);
    let errorMessage = error instanceof Error ? error.message : 'Crop failed';
    if (errorMessage.includes('Network request failed') || errorMessage.includes('fetch failed')) {
      errorMessage += `. Check if python CV service is running at ${CV_SERVICE_URL}`;
    }

    return {
      success: false,
      error: errorMessage,
      extractionMs: 0,
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Hybrid Extraction (CV + AI)
// ════════════════════════════════════════════════════════════════════════════

/**
 * Create AI prompt with CV-detected geometry as ground truth
 */
export function createGeometryConstrainedPrompt(cvResult: CVExtractionResult): string {
  const holePositions = findHolePositions(cvResult.shape);
  const holeDesc =
    holePositions.length > 0
      ? `Holes at positions: ${holePositions.map(([r, c]) => `(${r},${c})`).join(', ')}`
      : 'No holes detected';

  return `GEOMETRY (from pixel analysis - treat as ground truth):
- Grid: ${cvResult.rows} rows × ${cvResult.cols} columns
- Shape:
${cvResult.shape}
- ${holeDesc}
- ${cvResult.cells.length} cells detected

YOUR TASK: Only extract SEMANTIC information:
1. REGIONS: For each '.' cell in shape, identify its colored region (A-J)
2. CONSTRAINTS: Read numbers/symbols near regions (=12, <10, etc.)

Do NOT change the grid dimensions or hole positions - they are correct.

OUTPUT JSON:
{
  "regions": "<same dimensions as shape, letters for colors, # for holes>",
  "constraints": {"A": {"type": "sum", "op": "==", "value": 8}, ...},
  "confidence": {"regions": 0.9, "constraints": 0.9},
  "reasoning": "<brief explanation>"
}`;
}

/**
 * Find hole positions from shape string
 */
function findHolePositions(shape: string): [number, number][] {
  const holes: [number, number][] = [];
  const rows = shape.split('\n');
  for (let r = 0; r < rows.length; r++) {
    for (let c = 0; c < rows[r].length; c++) {
      if (rows[r][c] === '#') {
        holes.push([r, c]);
      }
    }
  }
  return holes;
}

/**
 * Validate that AI regions match CV shape
 */
export function validateRegionsMatchShape(shape: string, regions: string): string[] {
  const issues: string[] = [];
  const shapeRows = shape.split('\n');
  const regionRows = regions.split('\n');

  if (shapeRows.length !== regionRows.length) {
    issues.push(
      `Row count mismatch: shape has ${shapeRows.length}, regions has ${regionRows.length}`
    );
    return issues;
  }

  for (let r = 0; r < shapeRows.length; r++) {
    if (shapeRows[r].length !== regionRows[r].length) {
      issues.push(
        `Row ${r} length mismatch: shape has ${shapeRows[r].length}, regions has ${regionRows[r].length}`
      );
      continue;
    }

    for (let c = 0; c < shapeRows[r].length; c++) {
      const isHole = shapeRows[r][c] === '#';
      const regionChar = regionRows[r][c];

      if (isHole && regionChar !== '#') {
        issues.push(`Position (${r},${c}) should be hole '#' but got '${regionChar}'`);
      }
      if (!isHole && regionChar === '#') {
        issues.push(`Position (${r},${c}) has cell but regions shows hole '#'`);
      }
    }
  }

  return issues;
}

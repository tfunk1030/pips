/**
 * CV Extraction Client
 * Calls Python CV service for pixel-accurate grid geometry detection
 *
 * The CV service handles:
 * - Grid line detection
 * - Cell boundary detection
 * - Hole detection (critical for accuracy)
 * - Grid dimension inference
 * - Image preprocessing (contrast/brightness normalization)
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

// ────────────────────────────────────────────────────────────────────────────
// Image Preprocessing Types
// ────────────────────────────────────────────────────────────────────────────

/**
 * Options for image preprocessing before AI extraction
 */
export interface PreprocessOptions {
  /** Apply CLAHE for adaptive contrast enhancement (default: true) */
  normalizeContrast?: boolean;
  /** Normalize brightness levels (default: true) */
  normalizeBrightness?: boolean;
  /** Apply automatic white balance (default: true) */
  autoWhiteBalance?: boolean;
  /** Apply mild sharpening - useful for blurry images (default: false) */
  sharpen?: boolean;
  /** CLAHE contrast limit 1.0-4.0 (default: 2.0) */
  claheClipLimit?: number;
  /** CLAHE tile grid size 2-16 (default: 8) */
  claheGridSize?: number;
  /** Target mean brightness 0-255 (default: 128) */
  targetBrightness?: number;
  /** Tolerance for brightness adjustment (default: 30) */
  brightnessTolerance?: number;
}

/**
 * Image statistics for quality assessment
 */
export interface ImageStats {
  brightness: number;
  contrast: number;
  dynamicRange: number;
  minValue: number;
  maxValue: number;
  colorBalance: {
    red: number;
    green: number;
    blue: number;
  };
  saturation: number;
}

/**
 * Result of image preprocessing
 */
export interface PreprocessResult {
  success: boolean;
  error?: string;
  /** Preprocessed image as base64 PNG */
  preprocessedImage?: string;
  /** Original image statistics */
  originalStats?: ImageStats;
  /** Processed image statistics */
  processedStats?: ImageStats;
  /** List of operations that were applied */
  operationsApplied: string[];
  /** Processing time in milliseconds */
  extractionMs: number;
}

/**
 * Enhanced crop result with grid detection confidence
 */
export interface EnhancedCropResult {
  success: boolean;
  error?: string;
  croppedImage?: string;
  bounds?: {
    x: number;
    y: number;
    width: number;
    height: number;
    originalWidth: number;
    originalHeight: number;
  };
  gridBounds?: {
    x: number;
    y: number;
    width: number;
    height: number;
    originalWidth: number;
    originalHeight: number;
  };
  /** Grid detection confidence 0-1 */
  gridConfidence?: number;
  /** Categorical confidence level */
  confidenceLevel: 'high' | 'medium' | 'low' | 'unknown';
  /** User-facing warnings about detection quality */
  warnings: string[];
  /** Detected grid rows (if found via line detection) */
  detectedRows?: number;
  /** Detected grid columns (if found via line detection) */
  detectedCols?: number;
  /** Detection method used */
  detectionMethod: string;
  extractionMs: number;
}

/**
 * Result of full CV preprocessing pipeline (crop + preprocess)
 */
export interface CVPreprocessingResult {
  success: boolean;
  error?: string;
  /** Preprocessed and cropped puzzle image (base64) */
  puzzleImage?: string;
  /** Preprocessed and cropped domino tray image (base64) */
  dominoImage?: string;
  /** Grid detection info from cropping */
  gridInfo?: {
    confidence: number;
    confidenceLevel: 'high' | 'medium' | 'low' | 'unknown';
    detectedRows?: number;
    detectedCols?: number;
    warnings: string[];
  };
  /** Preprocessing info */
  preprocessingInfo?: {
    operationsApplied: string[];
    originalStats?: ImageStats;
    processedStats?: ImageStats;
  };
  /** Grid bounds for overlay alignment */
  gridBounds?: {
    x: number;
    y: number;
    width: number;
    height: number;
    originalWidth: number;
    originalHeight: number;
  };
  /** Total processing time */
  totalMs: number;
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
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Crop failed',
      extractionMs: 0,
    };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Image Preprocessing
// ════════════════════════════════════════════════════════════════════════════

/**
 * Default preprocessing options optimized for puzzle screenshots
 */
const DEFAULT_PREPROCESS_OPTIONS: Required<PreprocessOptions> = {
  normalizeContrast: true,
  normalizeBrightness: true,
  autoWhiteBalance: true,
  sharpen: false,
  claheClipLimit: 2.0,
  claheGridSize: 8,
  targetBrightness: 128,
  brightnessTolerance: 30,
};

/**
 * Transform snake_case API response to camelCase ImageStats
 */
function transformImageStats(apiStats: any): ImageStats | undefined {
  if (!apiStats) return undefined;
  return {
    brightness: apiStats.brightness,
    contrast: apiStats.contrast,
    dynamicRange: apiStats.dynamic_range,
    minValue: apiStats.min_value,
    maxValue: apiStats.max_value,
    colorBalance: apiStats.color_balance || { red: 0, green: 0, blue: 0 },
    saturation: apiStats.saturation,
  };
}

/**
 * Preprocess image using CV service for better AI extraction accuracy.
 *
 * Applies:
 * - CLAHE (Contrast Limited Adaptive Histogram Equalization)
 * - Brightness normalization
 * - Automatic white balance
 * - Optional sharpening
 *
 * @param base64Image - Input image as base64 string
 * @param options - Preprocessing options
 * @returns Preprocessed image with before/after statistics
 */
export async function preprocessImage(
  base64Image: string,
  options: PreprocessOptions = {}
): Promise<PreprocessResult> {
  const opts = { ...DEFAULT_PREPROCESS_OPTIONS, ...options };

  try {
    const startTime = Date.now();

    const response = await fetch(`${CV_SERVICE_URL}/preprocess-image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64Image,
        normalize_contrast: opts.normalizeContrast,
        normalize_brightness: opts.normalizeBrightness,
        auto_white_balance: opts.autoWhiteBalance,
        sharpen: opts.sharpen,
        clahe_clip_limit: opts.claheClipLimit,
        clahe_grid_size: opts.claheGridSize,
        target_brightness: opts.targetBrightness,
        brightness_tolerance: opts.brightnessTolerance,
      }),
    });

    if (!response.ok) {
      throw new Error(`CV service error: ${response.status}`);
    }

    const data = await response.json();
    const totalMs = Date.now() - startTime;

    if (!data.success) {
      return {
        success: false,
        error: data.error || 'Preprocessing failed',
        operationsApplied: [],
        extractionMs: data.extraction_ms || totalMs,
      };
    }

    return {
      success: true,
      preprocessedImage: data.preprocessed_image,
      originalStats: transformImageStats(data.original_stats),
      processedStats: transformImageStats(data.processed_stats),
      operationsApplied: data.operations_applied || [],
      extractionMs: data.extraction_ms || totalMs,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Preprocessing failed',
      operationsApplied: [],
      extractionMs: 0,
    };
  }
}

/**
 * Crop puzzle region with enhanced detection info.
 *
 * Uses the enhanced /crop-puzzle endpoint that provides:
 * - Grid detection confidence
 * - Detected grid dimensions
 * - Detection method used
 * - User-facing warnings
 *
 * @param base64Image - Input image as base64 string
 * @param options - Cropping options
 * @returns Enhanced crop result with detection info
 */
export async function cropPuzzleRegionEnhanced(
  base64Image: string,
  options: {
    excludeBottomPercent?: number;
    minConfidenceThreshold?: number;
    paddingPercent?: number;
  } = {}
): Promise<EnhancedCropResult> {
  try {
    const startTime = Date.now();

    const response = await fetch(`${CV_SERVICE_URL}/crop-puzzle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64Image,
        exclude_bottom_percent: options.excludeBottomPercent ?? 0.05,
        min_confidence_threshold: options.minConfidenceThreshold ?? 0.3,
        padding_percent: options.paddingPercent ?? 0.05,
      }),
    });

    if (!response.ok) {
      throw new Error(`CV service error: ${response.status}`);
    }

    const data = await response.json();
    const totalMs = Date.now() - startTime;

    if (!data.success) {
      return {
        success: false,
        error: data.error || 'Crop failed',
        confidenceLevel: 'unknown',
        warnings: data.warnings || [],
        detectionMethod: 'none',
        extractionMs: data.extraction_ms || totalMs,
      };
    }

    return {
      success: true,
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
      gridConfidence: data.grid_confidence,
      confidenceLevel: data.confidence_level || 'unknown',
      warnings: data.warnings || [],
      detectedRows: data.detected_rows,
      detectedCols: data.detected_cols,
      detectionMethod: data.detection_method || 'unknown',
      extractionMs: data.extraction_ms || totalMs,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Crop failed',
      confidenceLevel: 'unknown',
      warnings: [],
      detectionMethod: 'none',
      extractionMs: 0,
    };
  }
}

/**
 * Full CV preprocessing pipeline: crop + preprocess for optimal AI extraction.
 *
 * This function chains CV operations before AI extraction:
 * 1. Crop puzzle region (with grid detection)
 * 2. Crop domino tray region (if puzzle bottom detected)
 * 3. Apply image preprocessing to cropped regions
 *
 * The resulting images are cleaner and more consistent for AI analysis,
 * improving extraction accuracy especially for:
 * - Low-contrast images
 * - Images with color casts
 * - Images with varying lighting
 *
 * @param base64Image - Full screenshot as base64 string
 * @param options - Preprocessing options
 * @returns Preprocessed puzzle and domino images with grid detection info
 */
export async function preprocessForAIExtraction(
  base64Image: string,
  options: PreprocessOptions = {}
): Promise<CVPreprocessingResult> {
  const startTime = Date.now();

  try {
    // Step 1: Crop puzzle region with enhanced detection
    const cropResult = await cropPuzzleRegionEnhanced(base64Image);

    if (!cropResult.success || !cropResult.croppedImage) {
      return {
        success: false,
        error: cropResult.error || 'Failed to crop puzzle region',
        totalMs: Date.now() - startTime,
      };
    }

    // Step 2: Preprocess the cropped puzzle image
    const puzzlePreprocess = await preprocessImage(cropResult.croppedImage, options);

    if (!puzzlePreprocess.success || !puzzlePreprocess.preprocessedImage) {
      // Fall back to cropped but unprocessed image
      return {
        success: true,
        puzzleImage: cropResult.croppedImage,
        gridInfo: {
          confidence: cropResult.gridConfidence ?? 0,
          confidenceLevel: cropResult.confidenceLevel,
          detectedRows: cropResult.detectedRows,
          detectedCols: cropResult.detectedCols,
          warnings: [
            ...(cropResult.warnings || []),
            'Image preprocessing failed - using cropped image without enhancement',
          ],
        },
        gridBounds: cropResult.gridBounds,
        totalMs: Date.now() - startTime,
      };
    }

    // Step 3: Try to crop and preprocess domino region
    let dominoImage: string | undefined;
    const puzzleBottomY = cropResult.bounds
      ? cropResult.bounds.y + cropResult.bounds.height
      : undefined;

    if (puzzleBottomY !== undefined) {
      try {
        const dominoCrop = await cropDominoRegion(base64Image, puzzleBottomY);

        if (dominoCrop.success && dominoCrop.croppedImage) {
          // Preprocess domino image too
          const dominoPreprocess = await preprocessImage(dominoCrop.croppedImage, options);
          dominoImage = dominoPreprocess.success && dominoPreprocess.preprocessedImage
            ? dominoPreprocess.preprocessedImage
            : dominoCrop.croppedImage;
        }
      } catch {
        // Domino cropping is optional - continue without it
      }
    }

    return {
      success: true,
      puzzleImage: puzzlePreprocess.preprocessedImage,
      dominoImage,
      gridInfo: {
        confidence: cropResult.gridConfidence ?? 0,
        confidenceLevel: cropResult.confidenceLevel,
        detectedRows: cropResult.detectedRows,
        detectedCols: cropResult.detectedCols,
        warnings: cropResult.warnings || [],
      },
      preprocessingInfo: {
        operationsApplied: puzzlePreprocess.operationsApplied,
        originalStats: puzzlePreprocess.originalStats,
        processedStats: puzzlePreprocess.processedStats,
      },
      gridBounds: cropResult.gridBounds,
      totalMs: Date.now() - startTime,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'CV preprocessing failed',
      totalMs: Date.now() - startTime,
    };
  }
}

/**
 * Check if image needs preprocessing based on statistics.
 *
 * Returns true if the image has:
 * - Low contrast (std dev < 40)
 * - Poor brightness (too dark < 80 or too bright > 200)
 * - Low dynamic range (< 100)
 * - Color imbalance (channel difference > 30)
 *
 * @param stats - Image statistics from CV service
 * @returns Whether preprocessing is recommended
 */
export function shouldPreprocess(stats: ImageStats): boolean {
  // Low contrast
  if (stats.contrast < 40) return true;

  // Poor brightness
  if (stats.brightness < 80 || stats.brightness > 200) return true;

  // Low dynamic range
  if (stats.dynamicRange < 100) return true;

  // Significant color imbalance
  const { red, green, blue } = stats.colorBalance;
  const maxDiff = Math.max(
    Math.abs(red - green),
    Math.abs(green - blue),
    Math.abs(blue - red)
  );
  if (maxDiff > 30) return true;

  return false;
}

/**
 * Get preprocessing recommendations based on image statistics.
 *
 * Returns specific preprocessing options based on detected issues.
 *
 * @param stats - Image statistics from CV service
 * @returns Recommended preprocessing options
 */
export function getPreprocessingRecommendations(stats: ImageStats): PreprocessOptions {
  const options: PreprocessOptions = {};

  // Low contrast - use CLAHE with higher clip limit
  if (stats.contrast < 30) {
    options.normalizeContrast = true;
    options.claheClipLimit = 3.0;
  } else if (stats.contrast < 40) {
    options.normalizeContrast = true;
    options.claheClipLimit = 2.0;
  }

  // Poor brightness - normalize
  if (stats.brightness < 80 || stats.brightness > 200) {
    options.normalizeBrightness = true;
    options.targetBrightness = 128;
  }

  // Color imbalance - white balance
  const { red, green, blue } = stats.colorBalance;
  const maxDiff = Math.max(
    Math.abs(red - green),
    Math.abs(green - blue),
    Math.abs(blue - red)
  );
  if (maxDiff > 20) {
    options.autoWhiteBalance = true;
  }

  // Low dynamic range might benefit from sharpening
  if (stats.dynamicRange < 80) {
    options.sharpen = true;
  }

  return options;
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

/**
 * Multi-Stage Extraction Pipeline
 *
 * Orchestrates the 5-stage extraction process with 3-model ensemble.
 */

import {
  ExtractionConfig,
  ExtractionResult,
  StageName,
  StageConfidence,
  ValidationError,
} from './types';
import { DEFAULT_CONFIG } from './config';
import {
  extractGridGeometry,
  extractCellDetection,
  extractRegionMapping,
  extractConstraints,
  extractDominoes,
} from './stages';

// =============================================================================
// Progress Callback
// =============================================================================

export interface ExtractionProgress {
  stage: StageName;
  stageNumber: number;
  totalStages: number;
  status: 'starting' | 'complete' | 'retrying' | 'error';
  message: string;
  confidence?: number;
}

export type ProgressCallback = (progress: ExtractionProgress) => void;

// =============================================================================
// Main Pipeline Function
// =============================================================================

/**
 * Extract puzzle from image using 5-stage multi-model pipeline.
 *
 * Stages:
 * 1. Grid Geometry - Extract rows/cols
 * 2. Cell Detection - Identify cells vs holes
 * 3. Region Mapping - Map colored regions
 * 4. Constraint Extraction - Extract region constraints
 * 5. Domino Extraction - Extract domino tiles
 *
 * Each stage uses 3 models (Gemini 2.0 Flash, GPT-4o, Claude 3.5 Sonnet)
 * with confidence-weighted consensus.
 */
export async function extractPuzzle(
  imageBase64: string,
  config: Partial<ExtractionConfig> = {},
  onProgress?: ProgressCallback
): Promise<ExtractionResult> {
  // Merge with defaults
  const fullConfig: ExtractionConfig = {
    ...DEFAULT_CONFIG,
    ...config,
    apiKeys: { ...DEFAULT_CONFIG.apiKeys, ...config.apiKeys },
    models: { ...DEFAULT_CONFIG.models, ...config.models },
    validation: { ...DEFAULT_CONFIG.validation, ...config.validation },
  };

  const startTime = Date.now();
  let totalRetries = 0;
  const allValidationErrors: ValidationError[] = [];
  const reviewHints: string[] = [];

  // Normalize image base64 (remove data URL prefix if present)
  const normalizedImage = normalizeBase64(imageBase64);

  // ==========================================================================
  // Stage 1: Grid Geometry
  // ==========================================================================
  onProgress?.({
    stage: 'grid',
    stageNumber: 1,
    totalStages: 5,
    status: 'starting',
    message: 'Detecting grid dimensions...',
  });

  const gridResult = await extractGridGeometry(normalizedImage, fullConfig);
  totalRetries += gridResult.retryCount;

  if (gridResult.validationErrors.length > 0) {
    allValidationErrors.push(
      ...gridResult.validationErrors.map((msg) => ({
        stage: 'grid' as StageName,
        field: 'dimensions',
        message: msg,
      }))
    );
    reviewHints.push('Grid dimensions may be incorrect');
  }

  onProgress?.({
    stage: 'grid',
    stageNumber: 1,
    totalStages: 5,
    status: 'complete',
    message: `Grid: ${gridResult.result.rows}x${gridResult.result.cols}`,
    confidence: gridResult.result.confidence,
  });

  // ==========================================================================
  // Stage 2: Cell Detection
  // ==========================================================================
  onProgress?.({
    stage: 'cells',
    stageNumber: 2,
    totalStages: 5,
    status: 'starting',
    message: 'Detecting cells and holes...',
  });

  const cellResult = await extractCellDetection(
    normalizedImage,
    gridResult.result,
    fullConfig
  );
  totalRetries += cellResult.retryCount;

  if (cellResult.validationErrors.length > 0) {
    allValidationErrors.push(
      ...cellResult.validationErrors.map((msg) => ({
        stage: 'cells' as StageName,
        field: 'shape',
        message: msg,
      }))
    );
    reviewHints.push('Cell/hole detection may have errors');
  }

  const cellCount = (cellResult.result.shape.match(/\./g) || []).length;
  onProgress?.({
    stage: 'cells',
    stageNumber: 2,
    totalStages: 5,
    status: 'complete',
    message: `Found ${cellCount} cells`,
    confidence: cellResult.result.confidence,
  });

  // ==========================================================================
  // Stage 3: Region Mapping
  // ==========================================================================
  onProgress?.({
    stage: 'regions',
    stageNumber: 3,
    totalStages: 5,
    status: 'starting',
    message: 'Mapping colored regions...',
  });

  const regionResult = await extractRegionMapping(
    normalizedImage,
    gridResult.result,
    cellResult.result,
    fullConfig
  );
  totalRetries += regionResult.retryCount;

  if (regionResult.validationErrors.length > 0) {
    allValidationErrors.push(
      ...regionResult.validationErrors.map((msg) => ({
        stage: 'regions' as StageName,
        field: 'regions',
        message: msg,
      }))
    );
    reviewHints.push('Region boundaries may be incorrect');
  }

  // Count unique regions
  const uniqueRegions = new Set(
    regionResult.result.regions.replace(/[#\n]/g, '').split('')
  );
  onProgress?.({
    stage: 'regions',
    stageNumber: 3,
    totalStages: 5,
    status: 'complete',
    message: `Found ${uniqueRegions.size} regions`,
    confidence: regionResult.result.confidence,
  });

  // ==========================================================================
  // Stage 4: Constraint Extraction
  // ==========================================================================
  onProgress?.({
    stage: 'constraints',
    stageNumber: 4,
    totalStages: 5,
    status: 'starting',
    message: 'Extracting constraints...',
  });

  const constraintResult = await extractConstraints(
    normalizedImage,
    regionResult.result,
    fullConfig
  );
  totalRetries += constraintResult.retryCount;

  if (constraintResult.validationErrors.length > 0) {
    allValidationErrors.push(
      ...constraintResult.validationErrors.map((msg) => ({
        stage: 'constraints' as StageName,
        field: 'constraints',
        message: msg,
      }))
    );
    reviewHints.push('Constraints may need verification');
  }

  const constraintCount = Object.keys(constraintResult.result.constraints).length;
  onProgress?.({
    stage: 'constraints',
    stageNumber: 4,
    totalStages: 5,
    status: 'complete',
    message: `Found ${constraintCount} constraints`,
    confidence: constraintResult.result.confidence,
  });

  // ==========================================================================
  // Stage 5: Domino Extraction
  // ==========================================================================
  onProgress?.({
    stage: 'dominoes',
    stageNumber: 5,
    totalStages: 5,
    status: 'starting',
    message: 'Extracting dominoes...',
  });

  const dominoResult = await extractDominoes(
    normalizedImage,
    cellResult.result,
    fullConfig
  );
  totalRetries += dominoResult.retryCount;

  if (dominoResult.validationErrors.length > 0) {
    allValidationErrors.push(
      ...dominoResult.validationErrors.map((msg) => ({
        stage: 'dominoes' as StageName,
        field: 'dominoes',
        message: msg,
      }))
    );
    reviewHints.push('Domino extraction may have errors');
  }

  onProgress?.({
    stage: 'dominoes',
    stageNumber: 5,
    totalStages: 5,
    status: 'complete',
    message: `Found ${dominoResult.result.dominoes.length} dominoes`,
    confidence: dominoResult.result.confidence,
  });

  // ==========================================================================
  // Build Result
  // ==========================================================================
  const stageConfidence: StageConfidence = {
    grid: gridResult.result.confidence,
    cells: cellResult.result.confidence,
    regions: regionResult.result.confidence,
    constraints: constraintResult.result.confidence,
    dominoes: dominoResult.result.confidence,
  };

  const overallConfidence = Math.min(
    stageConfidence.grid,
    stageConfidence.cells,
    stageConfidence.regions,
    stageConfidence.constraints,
    stageConfidence.dominoes
  );

  const needsReview =
    overallConfidence < fullConfig.lowConfidenceThreshold ||
    allValidationErrors.length > 0;

  const result: ExtractionResult = {
    grid: {
      rows: gridResult.result.rows,
      cols: gridResult.result.cols,
      shape: cellResult.result.shape,
      regions: regionResult.result.regions,
    },
    constraints: constraintResult.result.constraints,
    dominoes: dominoResult.result.dominoes,
    confidence: {
      overall: overallConfidence,
      perStage: stageConfidence,
    },
    needsReview,
    reviewHints: [...new Set(reviewHints)], // Dedupe
  };

  // Add debug info if enabled
  if (fullConfig.saveDebugResponses) {
    result.debug = {
      rawResponses: {
        grid: gridResult.responses,
        cells: cellResult.responses,
        regions: regionResult.responses,
        constraints: constraintResult.responses,
        dominoes: dominoResult.responses,
      },
      consensusDetails: {
        grid: {
          answer: gridResult.result,
          source: 'confidence',
          confident: gridResult.result.confidence >= 0.7,
          responses: gridResult.responses,
          retryCount: gridResult.retryCount,
        },
        cells: {
          answer: cellResult.result,
          source: 'confidence',
          confident: cellResult.result.confidence >= 0.7,
          responses: cellResult.responses,
          retryCount: cellResult.retryCount,
        },
        regions: {
          answer: regionResult.result,
          source: 'confidence',
          confident: regionResult.result.confidence >= 0.7,
          responses: regionResult.responses,
          retryCount: regionResult.retryCount,
        },
        constraints: {
          answer: constraintResult.result,
          source: 'confidence',
          confident: constraintResult.result.confidence >= 0.7,
          responses: constraintResult.responses,
          retryCount: constraintResult.retryCount,
        },
        dominoes: {
          answer: dominoResult.result,
          source: 'confidence',
          confident: dominoResult.result.confidence >= 0.7,
          responses: dominoResult.responses,
          retryCount: dominoResult.retryCount,
        },
      },
      totalRetries,
      totalLatencyMs: Date.now() - startTime,
    };
  }

  return result;
}

// =============================================================================
// Helpers
// =============================================================================

/**
 * Normalize base64 image data by removing data URL prefix if present
 */
function normalizeBase64(imageBase64: string): string {
  // Remove data URL prefix if present
  const base64Match = imageBase64.match(/^data:image\/[^;]+;base64,(.+)$/);
  if (base64Match) {
    return base64Match[1];
  }

  // Remove any whitespace
  return imageBase64.replace(/\s/g, '');
}

// =============================================================================
// Legacy Fallback Wrapper
// =============================================================================

/**
 * Extract puzzle with fallback to legacy extraction on failure
 */
export async function extractPuzzleWithFallback(
  imageBase64: string,
  config: Partial<ExtractionConfig> = {},
  onProgress?: ProgressCallback,
  legacyExtractor?: (image: string) => Promise<ExtractionResult>
): Promise<ExtractionResult> {
  const fullConfig: ExtractionConfig = {
    ...DEFAULT_CONFIG,
    ...config,
  };

  try {
    return await extractPuzzle(imageBase64, config, onProgress);
  } catch (error) {
    if (fullConfig.enableLegacyFallback && legacyExtractor) {
      console.warn('Multi-stage extraction failed, using legacy extractor:', error);
      return await legacyExtractor(imageBase64);
    }
    throw error;
  }
}

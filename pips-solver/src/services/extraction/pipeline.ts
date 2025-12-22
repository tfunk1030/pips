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
  GridGeometryResult,
  CellDetectionResult,
  RegionMappingResult,
  ConstraintExtractionResult,
  DominoExtractionResult,
  ValidationResult,
} from './types';
import { DEFAULT_CONFIG } from './config';
import {
  extractGridGeometry,
  extractCellDetection,
  extractRegionMapping,
  extractConstraints,
  extractDominoes,
} from './stages';
import {
  validateGridGeometry,
  validateCellDetection,
  validateRegionMapping,
  validateConstraints,
  validateDominoes,
  checkDominoFeasibility,
} from './validation';
import { crossValidateGridAndCells } from './validation/gridValidator';

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
// Confidence Aggregation Types
// =============================================================================

/**
 * Weights for each extraction stage (must sum to 1.0)
 * Higher weights indicate more critical stages for overall accuracy
 */
const STAGE_WEIGHTS: StageConfidence = {
  grid: 0.25,      // Grid dimensions are foundational
  cells: 0.25,     // Cell/hole detection is critical for puzzle structure
  regions: 0.20,   // Region mapping affects constraint interpretation
  constraints: 0.15, // Constraints can be manually corrected more easily
  dominoes: 0.15,  // Dominoes can also be manually verified
};

/**
 * Penalty factors for validation issues
 */
const VALIDATION_PENALTY = {
  /** Per-error penalty applied to overall confidence */
  perError: 0.05,
  /** Maximum total penalty from errors */
  maxPenalty: 0.30,
  /** Penalty for critical errors (blocking issues) */
  criticalPenalty: 0.50,
};

/**
 * Thresholds for quality classification
 */
const QUALITY_THRESHOLDS = {
  excellent: 0.90,
  good: 0.75,
  acceptable: 0.60,
  poor: 0.40,
};

/**
 * Extended confidence breakdown including quality metrics
 */
export interface ConfidenceBreakdown {
  /** Weighted aggregate of all stage confidences */
  weightedAverage: number;
  /** Minimum confidence across all stages (bottleneck indicator) */
  minimum: number;
  /** Maximum confidence across all stages */
  maximum: number;
  /** Standard deviation of stage confidences (consistency measure) */
  standardDeviation: number;
  /** Penalty applied due to validation errors */
  validationPenalty: number;
  /** Final quality score after all adjustments */
  qualityScore: number;
  /** Human-readable quality classification */
  qualityClass: 'excellent' | 'good' | 'acceptable' | 'poor' | 'critical';
  /** Per-stage contribution to the weighted average */
  stageContributions: StageConfidence;
}

/**
 * Compute weighted average confidence from per-stage confidences
 */
function computeWeightedConfidence(perStage: StageConfidence): number {
  return (
    perStage.grid * STAGE_WEIGHTS.grid +
    perStage.cells * STAGE_WEIGHTS.cells +
    perStage.regions * STAGE_WEIGHTS.regions +
    perStage.constraints * STAGE_WEIGHTS.constraints +
    perStage.dominoes * STAGE_WEIGHTS.dominoes
  );
}

/**
 * Compute standard deviation of stage confidences (measure of consistency)
 */
function computeConfidenceStdDev(perStage: StageConfidence): number {
  const values = [
    perStage.grid,
    perStage.cells,
    perStage.regions,
    perStage.constraints,
    perStage.dominoes,
  ];
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const squaredDiffs = values.map((v) => Math.pow(v - mean, 2));
  return Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / values.length);
}

/**
 * Compute validation penalty based on error count and severity
 */
function computeValidationPenalty(
  errors: ValidationError[],
  hasCriticalError: boolean
): number {
  if (hasCriticalError) {
    return VALIDATION_PENALTY.criticalPenalty;
  }

  const errorPenalty = errors.length * VALIDATION_PENALTY.perError;
  return Math.min(errorPenalty, VALIDATION_PENALTY.maxPenalty);
}

/**
 * Classify quality based on score
 */
function classifyQuality(
  score: number
): 'excellent' | 'good' | 'acceptable' | 'poor' | 'critical' {
  if (score >= QUALITY_THRESHOLDS.excellent) return 'excellent';
  if (score >= QUALITY_THRESHOLDS.good) return 'good';
  if (score >= QUALITY_THRESHOLDS.acceptable) return 'acceptable';
  if (score >= QUALITY_THRESHOLDS.poor) return 'poor';
  return 'critical';
}

/**
 * Aggregate confidence scores from all stages into an overall quality score
 *
 * The aggregation follows the pattern from ensembleExtraction.ts:
 * - Uses weighted averaging based on stage importance
 * - Applies penalty for validation errors
 * - Returns detailed breakdown for debugging and user feedback
 */
function aggregateConfidence(
  perStage: StageConfidence,
  validationErrors: ValidationError[],
  hasCriticalError: boolean = false
): ConfidenceBreakdown {
  // Compute weighted average
  const weightedAverage = computeWeightedConfidence(perStage);

  // Compute min/max for range indication
  const values = [
    perStage.grid,
    perStage.cells,
    perStage.regions,
    perStage.constraints,
    perStage.dominoes,
  ];
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);

  // Compute standard deviation (consistency measure)
  const standardDeviation = computeConfidenceStdDev(perStage);

  // Compute validation penalty
  const validationPenalty = computeValidationPenalty(
    validationErrors,
    hasCriticalError
  );

  // Final quality score: weighted average minus penalty, clamped to [0, 1]
  const qualityScore = Math.max(0, Math.min(1, weightedAverage - validationPenalty));

  // Classify quality
  const qualityClass = classifyQuality(qualityScore);

  // Compute per-stage contributions (for debugging/transparency)
  const stageContributions: StageConfidence = {
    grid: perStage.grid * STAGE_WEIGHTS.grid,
    cells: perStage.cells * STAGE_WEIGHTS.cells,
    regions: perStage.regions * STAGE_WEIGHTS.regions,
    constraints: perStage.constraints * STAGE_WEIGHTS.constraints,
    dominoes: perStage.dominoes * STAGE_WEIGHTS.dominoes,
  };

  return {
    weightedAverage,
    minimum,
    maximum,
    standardDeviation,
    validationPenalty,
    qualityScore,
    qualityClass,
    stageContributions,
  };
}

/**
 * Generate review hints based on confidence breakdown
 */
function generateConfidenceHints(breakdown: ConfidenceBreakdown, perStage: StageConfidence): string[] {
  const hints: string[] = [];

  // Flag low-confidence stages
  const stages: [StageName, number][] = [
    ['grid', perStage.grid],
    ['cells', perStage.cells],
    ['regions', perStage.regions],
    ['constraints', perStage.constraints],
    ['dominoes', perStage.dominoes],
  ];

  for (const [stage, confidence] of stages) {
    if (confidence < QUALITY_THRESHOLDS.acceptable) {
      const stageNames: Record<StageName, string> = {
        grid: 'Grid dimensions',
        cells: 'Cell/hole detection',
        regions: 'Region mapping',
        constraints: 'Constraint extraction',
        dominoes: 'Domino extraction',
      };
      hints.push(`Low confidence in ${stageNames[stage]} (${(confidence * 100).toFixed(0)}%)`);
    }
  }

  // Flag high variance (inconsistent confidences)
  if (breakdown.standardDeviation > 0.15) {
    hints.push('Extraction confidence varies significantly across stages - verify all sections');
  }

  // Flag if minimum is dragging down overall score
  if (breakdown.minimum < QUALITY_THRESHOLDS.poor && breakdown.weightedAverage > QUALITY_THRESHOLDS.acceptable) {
    const bottleneck = stages.find(([, conf]) => conf === breakdown.minimum);
    if (bottleneck) {
      const stageNames: Record<StageName, string> = {
        grid: 'Grid dimensions',
        cells: 'Cell/hole detection',
        regions: 'Region mapping',
        constraints: 'Constraint extraction',
        dominoes: 'Domino extraction',
      };
      hints.push(`${stageNames[bottleneck[0]]} is the main source of uncertainty`);
    }
  }

  return hints;
}

// =============================================================================
// Inter-Stage Validation
// =============================================================================

/**
 * Result of inter-stage validation
 */
interface InterStageValidationResult {
  /** Whether the stage output is valid enough to proceed */
  canProceed: boolean;
  /** Validation errors encountered */
  errors: ValidationError[];
  /** Review hints for the user */
  hints: string[];
}

/**
 * Validate grid geometry output before proceeding to cell detection
 */
function validateGridOutput(grid: GridGeometryResult): InterStageValidationResult {
  const errors: ValidationError[] = [];
  const hints: string[] = [];

  const validationResult = validateGridGeometry(grid);

  if (!validationResult.valid) {
    errors.push(...validationResult.errors);
    hints.push('Grid dimensions may be incorrect - verify rows/columns');
  }

  // Can proceed even with warnings, but flag for review
  return {
    canProceed: !validationResult.errors.some((e) => e.message.includes('below minimum') || e.message.includes('above maximum')),
    errors,
    hints,
  };
}

/**
 * Validate cell detection output and cross-validate with grid
 */
function validateCellOutput(
  cells: CellDetectionResult,
  grid: GridGeometryResult
): InterStageValidationResult {
  const errors: ValidationError[] = [];
  const hints: string[] = [];

  // Basic cell validation
  const cellValidation = validateCellDetection(cells, grid);
  if (!cellValidation.valid) {
    errors.push(...cellValidation.errors);
    hints.push('Cell/hole detection may have errors');
  }

  // Cross-validate grid and cells
  const crossValidation = crossValidateGridAndCells(grid, cells);
  if (!crossValidation.valid) {
    errors.push(...crossValidation.errors);

    // Add specific hints based on error types
    for (const error of crossValidation.errors) {
      if (error.message.includes('disconnected')) {
        hints.push('Some cells appear disconnected - check for holes');
      }
      if (error.message.includes('density')) {
        hints.push('Unusual cell/hole ratio detected');
      }
      if (error.message.includes('confidence')) {
        hints.push('Confidence inconsistency between stages');
      }
      if (error.message.includes('Suspicious')) {
        hints.push('Unusual hole pattern detected - verify shape');
      }
    }
  }

  // Critical errors that prevent proceeding: dimension mismatch
  const hasCriticalError = errors.some(
    (e) =>
      e.message.includes("don't match grid") ||
      e.message.includes('expected') && e.field === 'shape' && !e.message.includes('even')
  );

  return {
    canProceed: !hasCriticalError,
    errors,
    hints: [...new Set(hints)], // Dedupe
  };
}

/**
 * Validate region mapping output against grid and cells
 */
function validateRegionOutput(
  regions: RegionMappingResult,
  grid: GridGeometryResult,
  cells: CellDetectionResult
): InterStageValidationResult {
  const errors: ValidationError[] = [];
  const hints: string[] = [];

  const validationResult = validateRegionMapping(regions, grid, cells);

  if (!validationResult.valid) {
    errors.push(...validationResult.errors);

    // Add specific hints
    for (const error of validationResult.errors) {
      if (error.message.includes('contiguous')) {
        hints.push('Region boundaries may be incorrect');
      }
      if (error.message.includes('hole')) {
        hints.push('Region/hole mismatch detected');
      }
      if (error.message.includes('only 1 cell')) {
        hints.push('Some regions are too small');
      }
    }
  }

  // Critical: dimension mismatch or holes not matching
  const hasCriticalError = errors.some(
    (e) =>
      e.message.includes('rows, expected') ||
      e.message.includes('cols, expected')
  );

  return {
    canProceed: !hasCriticalError,
    errors,
    hints: [...new Set(hints)],
  };
}

/**
 * Validate constraint output against regions
 */
function validateConstraintOutput(
  constraints: ConstraintExtractionResult,
  regions: RegionMappingResult
): InterStageValidationResult {
  const errors: ValidationError[] = [];
  const hints: string[] = [];

  const validationResult = validateConstraints(constraints, regions);

  if (!validationResult.valid) {
    errors.push(...validationResult.errors);

    for (const error of validationResult.errors) {
      if (error.message.includes('unknown region')) {
        hints.push('Constraint references unknown region');
      }
      if (error.message.includes('impossible')) {
        hints.push('Some constraints may be impossible to satisfy');
      }
      if (error.message.includes('invalid operator')) {
        hints.push('Invalid constraint operator detected');
      }
      if (error.message.includes('missing value')) {
        hints.push('Some sum constraints missing values');
      }
    }
  }

  // Constraints don't prevent proceeding - they can be manually corrected
  return {
    canProceed: true,
    errors,
    hints: [...new Set(hints)],
  };
}

/**
 * Validate domino output against cells and constraints
 */
function validateDominoOutput(
  dominoes: DominoExtractionResult,
  cells: CellDetectionResult,
  constraints: ConstraintExtractionResult
): InterStageValidationResult {
  const errors: ValidationError[] = [];
  const hints: string[] = [];

  // Basic domino validation
  const validationResult = validateDominoes(dominoes, cells);
  if (!validationResult.valid) {
    errors.push(...validationResult.errors);

    for (const error of validationResult.errors) {
      if (error.message.includes('expected')) {
        hints.push('Wrong number of dominoes detected');
      }
      if (error.message.includes('Duplicate')) {
        hints.push('Duplicate dominoes detected');
      }
      if (error.message.includes('outside range')) {
        hints.push('Invalid pip values detected');
      }
    }
  }

  // Feasibility check
  const cellCount = (cells.shape.match(/\./g) || []).length;
  const feasibility = checkDominoFeasibility(
    dominoes.dominoes,
    cellCount,
    constraints.constraints
  );

  if (!feasibility.feasible) {
    for (const issue of feasibility.issues) {
      errors.push({
        stage: 'dominoes',
        field: 'feasibility',
        message: issue,
      });
    }
    hints.push('Dominoes may not satisfy constraints');
  }

  return {
    canProceed: true, // Domino issues don't prevent completion
    errors,
    hints: [...new Set(hints)],
  };
}

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
 * Each stage uses 3 models (Gemini 3 Pro, GPT-5.2, Claude Opus 4.5)
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

  // Inter-stage validation: Grid Geometry
  const gridValidation = validateGridOutput(gridResult.result);
  allValidationErrors.push(...gridValidation.errors);
  reviewHints.push(...gridValidation.hints);

  if (!gridValidation.canProceed) {
    // Cannot proceed with invalid grid dimensions
    onProgress?.({
      stage: 'grid',
      stageNumber: 1,
      totalStages: 5,
      status: 'error',
      message: 'Grid validation failed - cannot proceed',
      confidence: gridResult.result.confidence,
    });

    // Return partial result with error info
    return {
      grid: {
        rows: gridResult.result.rows,
        cols: gridResult.result.cols,
        shape: '',
        regions: '',
      },
      constraints: {},
      dominoes: [],
      confidence: {
        overall: 0,
        perStage: {
          grid: gridResult.result.confidence,
          cells: 0,
          regions: 0,
          constraints: 0,
          dominoes: 0,
        },
      },
      needsReview: true,
      reviewHints: ['Grid validation failed: ' + gridValidation.errors.map((e) => e.message).join('; ')],
    };
  }

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

  // Inter-stage validation: Cell Detection + Cross-Validation with Grid
  const cellValidation = validateCellOutput(cellResult.result, gridResult.result);
  allValidationErrors.push(...cellValidation.errors);
  reviewHints.push(...cellValidation.hints);

  if (!cellValidation.canProceed) {
    // Cannot proceed with critically invalid cell detection
    onProgress?.({
      stage: 'cells',
      stageNumber: 2,
      totalStages: 5,
      status: 'error',
      message: 'Cell detection validation failed - cannot proceed',
      confidence: cellResult.result.confidence,
    });

    // Return partial result with error info
    return {
      grid: {
        rows: gridResult.result.rows,
        cols: gridResult.result.cols,
        shape: cellResult.result.shape,
        regions: '',
      },
      constraints: {},
      dominoes: [],
      confidence: {
        overall: 0,
        perStage: {
          grid: gridResult.result.confidence,
          cells: cellResult.result.confidence,
          regions: 0,
          constraints: 0,
          dominoes: 0,
        },
      },
      needsReview: true,
      reviewHints: ['Cell detection validation failed: ' + cellValidation.errors.map((e) => e.message).join('; ')],
    };
  }

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

  // Inter-stage validation: Region Mapping
  const regionValidation = validateRegionOutput(
    regionResult.result,
    gridResult.result,
    cellResult.result
  );
  allValidationErrors.push(...regionValidation.errors);
  reviewHints.push(...regionValidation.hints);

  if (!regionValidation.canProceed) {
    // Cannot proceed with critically invalid region mapping
    onProgress?.({
      stage: 'regions',
      stageNumber: 3,
      totalStages: 5,
      status: 'error',
      message: 'Region mapping validation failed - cannot proceed',
      confidence: regionResult.result.confidence,
    });

    // Return partial result with error info
    return {
      grid: {
        rows: gridResult.result.rows,
        cols: gridResult.result.cols,
        shape: cellResult.result.shape,
        regions: regionResult.result.regions,
      },
      constraints: {},
      dominoes: [],
      confidence: {
        overall: 0,
        perStage: {
          grid: gridResult.result.confidence,
          cells: cellResult.result.confidence,
          regions: regionResult.result.confidence,
          constraints: 0,
          dominoes: 0,
        },
      },
      needsReview: true,
      reviewHints: ['Region mapping validation failed: ' + regionValidation.errors.map((e) => e.message).join('; ')],
    };
  }

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

  // Inter-stage validation: Constraints
  const constraintValidation = validateConstraintOutput(
    constraintResult.result,
    regionResult.result
  );
  allValidationErrors.push(...constraintValidation.errors);
  reviewHints.push(...constraintValidation.hints);

  // Constraints don't block proceeding - issues are flagged for user review

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

  // Inter-stage validation: Dominoes (final validation with feasibility check)
  const dominoValidation = validateDominoOutput(
    dominoResult.result,
    cellResult.result,
    constraintResult.result
  );
  allValidationErrors.push(...dominoValidation.errors);
  reviewHints.push(...dominoValidation.hints);

  // Dominoes don't block completion - issues are flagged for user review

  // ==========================================================================
  // Build Result with Confidence Aggregation
  // ==========================================================================
  const stageConfidence: StageConfidence = {
    grid: gridResult.result.confidence,
    cells: cellResult.result.confidence,
    regions: regionResult.result.confidence,
    constraints: constraintResult.result.confidence,
    dominoes: dominoResult.result.confidence,
  };

  // Use weighted confidence aggregation for more accurate quality scoring
  // This follows the pattern from ensembleExtraction.ts
  const confidenceBreakdown = aggregateConfidence(
    stageConfidence,
    allValidationErrors,
    false // No critical errors if we've reached this point
  );

  // Generate additional hints based on confidence analysis
  const confidenceHints = generateConfidenceHints(confidenceBreakdown, stageConfidence);
  reviewHints.push(...confidenceHints);

  // Use the quality score as the overall confidence
  // This is more nuanced than the previous Math.min approach
  const overallConfidence = confidenceBreakdown.qualityScore;

  // Determine if review is needed based on quality classification
  const needsReview =
    confidenceBreakdown.qualityClass === 'poor' ||
    confidenceBreakdown.qualityClass === 'critical' ||
    overallConfidence < fullConfig.lowConfidenceThreshold ||
    allValidationErrors.length > 0;

  const result: ExtractionResult = {
    grid: {
      rows: gridResult.result.rows,
      cols: gridResult.result.cols,
      shape: cellResult.result.shape,
      regions: regionResult.result.regions,
      gridLocation: gridResult.result.gridLocation,
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

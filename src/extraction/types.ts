/**
 * Extraction pipeline types for board image processing and confidence tracking.
 */

/**
 * Stages in the extraction pipeline, executed in order.
 */
export type ExtractionStage =
  | 'BOARD_DETECTION'
  | 'GRID_ALIGNMENT'
  | 'CELL_EXTRACTION'
  | 'PIP_RECOGNITION';

/**
 * Severity levels for confidence hints.
 */
export type HintSeverity = 'low' | 'medium' | 'high';

/**
 * Confidence score for a single extraction stage.
 */
export interface StageConfidence {
  stage: ExtractionStage;
  /** Confidence score from 0 to 1 */
  confidence: number;
  /** Method used for this stage (e.g., 'edge_detection', 'template_matching') */
  method: string;
  /** Optional additional details about the confidence calculation */
  details?: Record<string, unknown>;
}

/**
 * Human-readable hint about confidence issues at a specific stage.
 */
export interface ConfidenceHint {
  stage: ExtractionStage;
  severity: HintSeverity;
  /** User-facing message describing the issue (e.g., 'Low confidence in Grid dimensions') */
  message: string;
  /** Actionable suggestion to improve extraction (e.g., 'Try capturing with better lighting') */
  suggestion: string;
}

/**
 * Cell value extracted from the puzzle grid.
 */
export interface ExtractedCell {
  row: number;
  col: number;
  /** Pip value (0-6 typically, or null if uncertain) */
  value: number | null;
  /** Confidence in this specific cell's extraction */
  confidence: number;
}

/**
 * Complete result of the extraction pipeline.
 */
export interface ExtractionResult {
  /** Whether extraction completed successfully */
  success: boolean;
  /** Grid dimensions */
  rows: number;
  cols: number;
  /** Extracted cell values */
  cells: ExtractedCell[];
  /** Confidence scores for each pipeline stage */
  stageConfidences: StageConfidence[];
  /** Human-readable hints about confidence issues */
  hints: ConfidenceHint[];
  /** Overall extraction confidence (0-1, weighted average of stage confidences) */
  overallConfidence: number;
  /** Processing time in milliseconds */
  processingTimeMs: number;
  /** Error message if extraction failed */
  error?: string;
}

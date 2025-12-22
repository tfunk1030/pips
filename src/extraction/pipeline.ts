/**
 * Extraction pipeline functions for board image processing and confidence analysis.
 */

import {
  ExtractionStage,
  StageConfidence,
  ConfidenceHint,
  HintSeverity,
} from './types';

/** Threshold below which a stage confidence is considered low */
const LOW_CONFIDENCE_THRESHOLD = 0.7;

/** Threshold above which variance across stages triggers a warning */
const HIGH_VARIANCE_THRESHOLD = 0.15;

/** Human-readable names for each extraction stage */
const STAGE_DISPLAY_NAMES: Record<ExtractionStage, string> = {
  BOARD_DETECTION: 'Board detection',
  GRID_ALIGNMENT: 'Grid dimensions',
  CELL_EXTRACTION: 'Cell extraction',
  PIP_RECOGNITION: 'Pip recognition',
};

/** Suggestions for improving confidence at each stage */
const STAGE_SUGGESTIONS: Record<ExtractionStage, string> = {
  BOARD_DETECTION: 'Ensure the entire board is visible and centered in the frame',
  GRID_ALIGNMENT: 'Try capturing from directly above the board with less angle',
  CELL_EXTRACTION: 'Ensure good lighting and avoid shadows across the board',
  PIP_RECOGNITION: 'Make sure pip markings are clearly visible and not worn',
};

/**
 * Calculate the variance of confidence values.
 */
function calculateVariance(confidences: number[]): number {
  if (confidences.length === 0) return 0;
  const mean = confidences.reduce((sum, c) => sum + c, 0) / confidences.length;
  const squaredDiffs = confidences.map((c) => (c - mean) ** 2);
  return squaredDiffs.reduce((sum, d) => sum + d, 0) / confidences.length;
}

/**
 * Determine severity based on confidence value.
 */
function getSeverityForConfidence(confidence: number): HintSeverity {
  if (confidence < 0.5) return 'high';
  if (confidence < 0.7) return 'medium';
  return 'low';
}

/**
 * Analyze stage confidences and generate human-readable warning hints.
 *
 * Generates hints for:
 * - Individual stages with low confidence (< 0.7)
 * - High variance across stages (variance > 0.15)
 *
 * @param stageConfidences Array of confidence scores for each pipeline stage
 * @returns Array of human-readable hints about confidence issues
 */
export function generateConfidenceHints(
  stageConfidences: StageConfidence[],
): ConfidenceHint[] {
  const hints: ConfidenceHint[] = [];

  if (stageConfidences.length === 0) {
    return hints;
  }

  // Check for low confidence in individual stages
  for (const sc of stageConfidences) {
    if (sc.confidence < LOW_CONFIDENCE_THRESHOLD) {
      const stageName = STAGE_DISPLAY_NAMES[sc.stage];
      hints.push({
        stage: sc.stage,
        severity: getSeverityForConfidence(sc.confidence),
        message: `Low confidence in ${stageName}`,
        suggestion: STAGE_SUGGESTIONS[sc.stage],
      });
    }
  }

  // Check for high variance across stages
  const confidenceValues = stageConfidences.map((sc) => sc.confidence);
  const variance = calculateVariance(confidenceValues);

  if (variance > HIGH_VARIANCE_THRESHOLD) {
    // Find the stage with lowest confidence to attach this hint to
    const lowestStage = stageConfidences.reduce((min, sc) =>
      sc.confidence < min.confidence ? sc : min,
    );

    hints.push({
      stage: lowestStage.stage,
      severity: 'medium',
      message: 'Extraction confidence varies significantly across stages',
      suggestion: 'Some stages performed much better than others. Review the extraction results carefully.',
    });
  }

  return hints;
}

/**
 * Calculate the overall confidence from stage confidences.
 * Uses weighted average, giving more weight to later stages as they depend on earlier ones.
 *
 * @param stageConfidences Array of confidence scores for each pipeline stage
 * @returns Overall confidence score from 0 to 1
 */
export function calculateOverallConfidence(
  stageConfidences: StageConfidence[],
): number {
  if (stageConfidences.length === 0) return 0;

  // Simple average for now; could be weighted by stage importance
  const sum = stageConfidences.reduce((acc, sc) => acc + sc.confidence, 0);
  return sum / stageConfidences.length;
}

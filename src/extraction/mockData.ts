/**
 * Mock extraction results for development and testing.
 * Provides sample data with various confidence levels to test UI components.
 */

import {
  ExtractionStage,
  StageConfidence,
  ExtractedCell,
  ExtractionResult,
} from './types';
import { generateConfidenceHints, calculateOverallConfidence } from './pipeline';

/**
 * Generate a simple grid of extracted cells for testing.
 */
function generateMockCells(rows: number, cols: number, baseConfidence: number): ExtractedCell[] {
  const cells: ExtractedCell[] = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      cells.push({
        row,
        col,
        value: Math.floor(Math.random() * 7), // Random pip value 0-6
        confidence: baseConfidence + (Math.random() * 0.1 - 0.05), // Small variance around base
      });
    }
  }
  return cells;
}

/**
 * Create a complete ExtractionResult from stage confidences.
 */
function createExtractionResult(
  stageConfidences: StageConfidence[],
  rows: number = 5,
  cols: number = 5,
): ExtractionResult {
  const overallConfidence = calculateOverallConfidence(stageConfidences);
  const hints = generateConfidenceHints(stageConfidences);

  return {
    success: true,
    rows,
    cols,
    cells: generateMockCells(rows, cols, overallConfidence),
    stageConfidences,
    hints,
    overallConfidence,
    processingTimeMs: Math.floor(Math.random() * 500) + 200,
  };
}

/**
 * High confidence scenario - all stages > 0.9
 * Expected: No hints generated, high overall confidence
 */
export const highConfidenceResult: ExtractionResult = createExtractionResult([
  {
    stage: 'BOARD_DETECTION',
    confidence: 0.95,
    method: 'edge_detection',
    details: { edgeCount: 4, cornerDetection: 'strong' },
  },
  {
    stage: 'GRID_ALIGNMENT',
    confidence: 0.92,
    method: 'hough_transform',
    details: { linesDetected: 12, gridFit: 'excellent' },
  },
  {
    stage: 'CELL_EXTRACTION',
    confidence: 0.94,
    method: 'contour_detection',
    details: { cellsFound: 25, uniformSize: true },
  },
  {
    stage: 'PIP_RECOGNITION',
    confidence: 0.91,
    method: 'template_matching',
    details: { templatesMatched: 25, avgMatchScore: 0.91 },
  },
]);

/**
 * Low single-stage confidence scenario - one stage performs poorly
 * Expected: Single hint for the low-confidence stage
 */
export const lowSingleStageResult: ExtractionResult = createExtractionResult([
  {
    stage: 'BOARD_DETECTION',
    confidence: 0.88,
    method: 'edge_detection',
    details: { edgeCount: 4, cornerDetection: 'good' },
  },
  {
    stage: 'GRID_ALIGNMENT',
    confidence: 0.55,
    method: 'hough_transform',
    details: { linesDetected: 8, gridFit: 'poor', note: 'Perspective distortion detected' },
  },
  {
    stage: 'CELL_EXTRACTION',
    confidence: 0.82,
    method: 'contour_detection',
    details: { cellsFound: 25, uniformSize: false },
  },
  {
    stage: 'PIP_RECOGNITION',
    confidence: 0.79,
    method: 'template_matching',
    details: { templatesMatched: 22, avgMatchScore: 0.79 },
  },
]);

/**
 * High variance across stages scenario - significant difference between best and worst
 * Expected: Variance warning plus individual stage warnings
 */
export const highVarianceResult: ExtractionResult = createExtractionResult([
  {
    stage: 'BOARD_DETECTION',
    confidence: 0.98,
    method: 'edge_detection',
    details: { edgeCount: 4, cornerDetection: 'excellent' },
  },
  {
    stage: 'GRID_ALIGNMENT',
    confidence: 0.95,
    method: 'hough_transform',
    details: { linesDetected: 12, gridFit: 'excellent' },
  },
  {
    stage: 'CELL_EXTRACTION',
    confidence: 0.45,
    method: 'contour_detection',
    details: { cellsFound: 20, uniformSize: false, note: 'Shadow interference' },
  },
  {
    stage: 'PIP_RECOGNITION',
    confidence: 0.52,
    method: 'template_matching',
    details: { templatesMatched: 15, avgMatchScore: 0.52, note: 'Worn pip markings' },
  },
]);

/**
 * Mixed scenario - multiple issues across different stages
 * Expected: Multiple hints for various low-confidence stages
 */
export const mixedIssuesResult: ExtractionResult = createExtractionResult([
  {
    stage: 'BOARD_DETECTION',
    confidence: 0.65,
    method: 'edge_detection',
    details: { edgeCount: 3, cornerDetection: 'partial', note: 'Board partially obscured' },
  },
  {
    stage: 'GRID_ALIGNMENT',
    confidence: 0.58,
    method: 'hough_transform',
    details: { linesDetected: 9, gridFit: 'poor', note: 'Significant angle detected' },
  },
  {
    stage: 'CELL_EXTRACTION',
    confidence: 0.72,
    method: 'contour_detection',
    details: { cellsFound: 24, uniformSize: false },
  },
  {
    stage: 'PIP_RECOGNITION',
    confidence: 0.68,
    method: 'template_matching',
    details: { templatesMatched: 21, avgMatchScore: 0.68 },
  },
]);

/**
 * Failed extraction scenario - extraction did not complete successfully
 * Expected: Error state with no cell data
 */
export const failedExtractionResult: ExtractionResult = {
  success: false,
  rows: 0,
  cols: 0,
  cells: [],
  stageConfidences: [
    {
      stage: 'BOARD_DETECTION',
      confidence: 0.25,
      method: 'edge_detection',
      details: { edgeCount: 1, cornerDetection: 'failed' },
    },
  ],
  hints: generateConfidenceHints([
    {
      stage: 'BOARD_DETECTION',
      confidence: 0.25,
      method: 'edge_detection',
    },
  ]),
  overallConfidence: 0.25,
  processingTimeMs: 150,
  error: 'Failed to detect board boundaries. Ensure the board is fully visible in the frame.',
};

/**
 * All available mock scenarios for testing.
 */
export const mockScenarios = {
  highConfidence: highConfidenceResult,
  lowSingleStage: lowSingleStageResult,
  highVariance: highVarianceResult,
  mixedIssues: mixedIssuesResult,
  failed: failedExtractionResult,
} as const;

/**
 * Get a random mock scenario for quick testing.
 */
export function getRandomMockResult(): ExtractionResult {
  const keys = Object.keys(mockScenarios) as (keyof typeof mockScenarios)[];
  const randomKey = keys[Math.floor(Math.random() * keys.length)];
  return mockScenarios[randomKey];
}

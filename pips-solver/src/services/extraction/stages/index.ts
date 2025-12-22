/**
 * Extraction Stages
 *
 * Re-exports all stage extraction functions.
 */

export { extractGridGeometry, type GridGeometryStageResult } from './gridGeometry';
export { extractCellDetection, type CellDetectionStageResult } from './cellDetection';
export { extractRegionMapping, type RegionMappingStageResult } from './regionMapping';
export { extractConstraints, type ConstraintStageResult } from './constraintExtraction';
export { extractDominoes, type DominoStageResult } from './dominoExtraction';

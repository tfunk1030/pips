/**
 * Multi-Stage Extraction Pipeline
 *
 * Main entry point for the 5-stage extraction pipeline with 3-model ensemble.
 */

// Re-export types
export * from './types';

// Re-export config
export {
  DEFAULT_CONFIG,
  DEFAULT_MODELS,
  NYT_VALIDATION,
  createConfig,
  createOpenRouterConfig,
  createDirectConfig,
  hasValidApiKeys,
  getAvailableModels,
  hasEnsembleCapability,
} from './config';

// Re-export API client
export {
  callVisionApi,
  callAllModels,
  callWithRetry,
  resolveApiEndpoint,
} from './apiClient';

// Re-export consensus
export {
  resolveConsensus,
  resolveGridConsensus,
  resolveShapeConsensus,
  resolveDominoConsensus,
  shouldRetry,
  mergeConstraints,
  mergeDominoes,
} from './consensus';

// Re-export pipeline
export {
  extractPuzzle,
  extractPuzzleWithFallback,
  type ExtractionProgress,
  type ProgressCallback,
} from './pipeline';

// Re-export stages
export {
  extractGridGeometry,
  extractCellDetection,
  extractRegionMapping,
  extractConstraints,
  extractDominoes,
} from './stages';

// Re-export validation
export {
  validateExtractionResult,
  validateGridGeometry,
  validateCellDetection,
  validateRegionMapping,
  validateConstraints,
  validateDominoes,
  checkDominoFeasibility,
  getDominoStats,
  checkRotationMismatch,
} from './validation';

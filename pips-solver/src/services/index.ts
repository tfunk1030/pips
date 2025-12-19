/**
 * Services Index
 * Export all service functions and types
 */

// AI Extraction (main entry point)
export {
  convertAIResultToBuilderState,
  extractPuzzleFromImage,
  extractPuzzleMaxAccuracy,
  extractPuzzleMultiModel,
  type ExtractionProgress,
  type MultiModelExtractionOptions,
  type MultiModelExtractionResult,
} from './aiExtraction';

// Ensemble Extraction (internal, but exportable for advanced use)
export {
  extractPuzzleEnsemble,
  type EnsembleExtractionOptions,
  type EnsembleExtractionResult,
  type ExtractionProgress as EnsembleProgress,
} from './ensembleExtraction';

// Model Clients (for direct API access)
export {
  callMultipleModels,
  callVisionModel,
  createImageMessage,
  inferMediaType,
  normalizeBase64,
  type APIKeys,
  type ModelClientOptions,
  type VisionContent,
  type VisionMessage,
  type VisionResponse,
} from './modelClients';

/**
 * Hook for running the extraction pipeline with loading states.
 * Wraps the extraction pipeline functions and provides a clean interface for UI components.
 */

import { useState, useCallback } from 'react';
import { ExtractionResult, StageConfidence } from '../../extraction/types';
import { generateConfidenceHints, calculateOverallConfidence } from '../../extraction/pipeline';
import { getRandomMockResult } from '../../extraction/mockData';

export interface UseExtractionState {
  /** Whether extraction is currently in progress */
  isExtracting: boolean;
  /** The result of the last extraction, or null if not yet extracted */
  result: ExtractionResult | null;
  /** Error message if extraction failed */
  error: string | null;
}

export interface UseExtractionReturn extends UseExtractionState {
  /** Start extraction from an image URI */
  extractFromImage: (imageUri: string) => Promise<ExtractionResult>;
  /** Retry the last extraction */
  retry: () => Promise<ExtractionResult | null>;
  /** Clear the current result */
  clear: () => void;
}

/**
 * Simulate extraction processing delay.
 * In production, this would be replaced with actual image processing.
 */
async function simulateProcessingDelay(minMs: number = 500, maxMs: number = 1500): Promise<void> {
  const delay = Math.floor(Math.random() * (maxMs - minMs)) + minMs;
  return new Promise((resolve) => setTimeout(resolve, delay));
}

/**
 * Hook for running the extraction pipeline.
 *
 * Provides loading states and wraps the extraction pipeline functions.
 * Currently uses mock data for development - in production this would
 * integrate with actual image processing logic.
 *
 * @example
 * ```tsx
 * const { isExtracting, result, extractFromImage, retry } = useExtraction();
 *
 * const handleCapture = async (uri: string) => {
 *   const result = await extractFromImage(uri);
 *   navigation.navigate('ExtractionResult', { extractionResult: result });
 * };
 * ```
 */
export function useExtraction(): UseExtractionReturn {
  const [state, setState] = useState<UseExtractionState>({
    isExtracting: false,
    result: null,
    error: null,
  });

  const [lastImageUri, setLastImageUri] = useState<string | null>(null);

  /**
   * Run extraction on an image.
   *
   * Currently uses mock data for development. In production, this would:
   * 1. Load the image from the URI
   * 2. Run board detection
   * 3. Run grid alignment
   * 4. Run cell extraction
   * 5. Run pip recognition
   * 6. Generate confidence hints from stage results
   */
  const extractFromImage = useCallback(async (imageUri: string): Promise<ExtractionResult> => {
    setState((prev) => ({
      ...prev,
      isExtracting: true,
      error: null,
    }));

    setLastImageUri(imageUri);

    try {
      // Simulate processing delay
      await simulateProcessingDelay();

      // Get mock result (in production, this would be actual extraction)
      const mockResult = getRandomMockResult();

      // The mock result already has hints generated, but we could regenerate
      // them here to ensure consistency with the latest pipeline logic:
      const freshHints = generateConfidenceHints(mockResult.stageConfidences);
      const overallConfidence = calculateOverallConfidence(mockResult.stageConfidences);

      const result: ExtractionResult = {
        ...mockResult,
        hints: freshHints,
        overallConfidence,
      };

      setState({
        isExtracting: false,
        result,
        error: null,
      });

      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Extraction failed';
      setState({
        isExtracting: false,
        result: null,
        error: errorMessage,
      });

      // Return a failed extraction result
      const failedResult: ExtractionResult = {
        success: false,
        rows: 0,
        cols: 0,
        cells: [],
        stageConfidences: [],
        hints: [],
        overallConfidence: 0,
        processingTimeMs: 0,
        error: errorMessage,
      };

      return failedResult;
    }
  }, []);

  /**
   * Retry extraction with the last used image URI.
   * Returns null if no previous image URI exists.
   */
  const retry = useCallback(async (): Promise<ExtractionResult | null> => {
    if (!lastImageUri) {
      return null;
    }
    return extractFromImage(lastImageUri);
  }, [lastImageUri, extractFromImage]);

  /**
   * Clear the current extraction result.
   */
  const clear = useCallback(() => {
    setState({
      isExtracting: false,
      result: null,
      error: null,
    });
    setLastImageUri(null);
  }, []);

  return {
    ...state,
    extractFromImage,
    retry,
    clear,
  };
}

export default useExtraction;

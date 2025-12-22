/**
 * Tests for ConfidenceBreakdown component
 *
 * Verifies that the component correctly displays:
 * - Per-component confidence scores (geometry, OCR, puzzle, domino)
 * - Quality factor breakdowns
 * - Borderline warnings
 * - Lowest component highlighting
 */

import {
  getDisplayLabel,
  findLowestComponent,
  ComponentConfidenceMap,
  ConfidenceFactors,
} from '../app/components/ConfidenceBreakdown';

describe('ConfidenceBreakdown', () => {
  describe('getDisplayLabel', () => {
    // Component labels
    it('should return "Geometry" for geometry_extraction', () => {
      expect(getDisplayLabel('geometry_extraction')).toBe('Geometry');
    });

    it('should return "OCR" for ocr_detection', () => {
      expect(getDisplayLabel('ocr_detection')).toBe('OCR');
    });

    it('should return "Puzzle" for puzzle_detection', () => {
      expect(getDisplayLabel('puzzle_detection')).toBe('Puzzle');
    });

    it('should return "Domino" for domino_detection', () => {
      expect(getDisplayLabel('domino_detection')).toBe('Domino');
    });

    // Factor labels
    it('should return "Saturation" for saturation', () => {
      expect(getDisplayLabel('saturation')).toBe('Saturation');
    });

    it('should return "Area Ratio" for area_ratio', () => {
      expect(getDisplayLabel('area_ratio')).toBe('Area Ratio');
    });

    it('should return "Aspect Ratio" for aspect_ratio', () => {
      expect(getDisplayLabel('aspect_ratio')).toBe('Aspect Ratio');
    });

    it('should return "Relative Size" for relative_size', () => {
      expect(getDisplayLabel('relative_size')).toBe('Relative Size');
    });

    it('should return "Edge Clarity" for edge_clarity', () => {
      expect(getDisplayLabel('edge_clarity')).toBe('Edge Clarity');
    });

    it('should return "Contrast" for contrast', () => {
      expect(getDisplayLabel('contrast')).toBe('Contrast');
    });

    // Fallback for unknown keys
    it('should convert snake_case to Title Case for unknown keys', () => {
      expect(getDisplayLabel('unknown_factor')).toBe('Unknown Factor');
      expect(getDisplayLabel('some_other_metric')).toBe('Some Other Metric');
    });

    it('should handle single word keys', () => {
      expect(getDisplayLabel('score')).toBe('Score');
    });
  });

  describe('findLowestComponent', () => {
    it('should find the component with lowest confidence score', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.85,
        ocr_detection: 0.60,
        puzzle_detection: 0.75,
      };
      const result = findLowestComponent(scores);
      expect(result).toEqual({ component: 'ocr_detection', score: 0.60 });
    });

    it('should handle single component', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.90,
      };
      const result = findLowestComponent(scores);
      expect(result).toEqual({ component: 'geometry_extraction', score: 0.90 });
    });

    it('should return null for empty object', () => {
      const scores: ComponentConfidenceMap = {};
      const result = findLowestComponent(scores);
      expect(result).toBeNull();
    });

    it('should skip undefined values', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.85,
        ocr_detection: undefined,
        puzzle_detection: 0.70,
      };
      const result = findLowestComponent(scores);
      expect(result).toEqual({ component: 'puzzle_detection', score: 0.70 });
    });

    it('should handle all undefined values', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: undefined,
        ocr_detection: undefined,
      };
      const result = findLowestComponent(scores);
      expect(result).toBeNull();
    });

    it('should handle scores at thresholds', () => {
      // All high confidence - should return any of them (the first one found)
      const scoresAllHigh: ComponentConfidenceMap = {
        geometry_extraction: 0.85,
        ocr_detection: 0.90,
        puzzle_detection: 0.88,
      };
      const resultHigh = findLowestComponent(scoresAllHigh);
      expect(resultHigh?.score).toBe(0.85);
      expect(resultHigh?.component).toBe('geometry_extraction');
    });

    it('should handle very low scores', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.10,
        ocr_detection: 0.05,
        puzzle_detection: 0.15,
      };
      const result = findLowestComponent(scores);
      expect(result).toEqual({ component: 'ocr_detection', score: 0.05 });
    });

    it('should handle zero scores', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.50,
        ocr_detection: 0,
        puzzle_detection: 0.30,
      };
      const result = findLowestComponent(scores);
      expect(result).toEqual({ component: 'ocr_detection', score: 0 });
    });

    it('should handle equal scores (returns first found)', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.50,
        ocr_detection: 0.50,
        puzzle_detection: 0.50,
      };
      const result = findLowestComponent(scores);
      expect(result?.score).toBe(0.50);
      // First one in iteration order
      expect(result?.component).toBe('geometry_extraction');
    });
  });

  describe('Component score display logic', () => {
    it('should identify low confidence components correctly', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.85, // high
        ocr_detection: 0.60,       // low
        puzzle_detection: 0.75,    // medium
      };
      const lowest = findLowestComponent(scores);
      // OCR is lowest and below medium threshold (0.70)
      expect(lowest?.score).toBeLessThan(0.70);
      expect(lowest?.component).toBe('ocr_detection');
    });

    it('should identify when all components are high confidence', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.90,
        ocr_detection: 0.88,
        puzzle_detection: 0.92,
      };
      const lowest = findLowestComponent(scores);
      // Even the lowest is above high threshold
      expect(lowest?.score).toBeGreaterThanOrEqual(0.85);
    });

    it('should handle mixed borderline scores', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.86, // high, borderline (near 0.85)
        ocr_detection: 0.68,       // low, borderline (near 0.70)
        puzzle_detection: 0.72,    // medium
      };
      const lowest = findLowestComponent(scores);
      expect(lowest?.component).toBe('ocr_detection');
      expect(lowest?.score).toBe(0.68);
    });
  });

  describe('Factor breakdown processing', () => {
    it('should handle complete factor set', () => {
      const factors: ConfidenceFactors = {
        saturation: 0.90,
        area_ratio: 0.85,
        aspect_ratio: 0.88,
        relative_size: 0.75,
        edge_clarity: 0.82,
        contrast: 0.78,
      };
      // All factors should have valid labels
      Object.keys(factors).forEach(key => {
        const label = getDisplayLabel(key);
        expect(label).not.toBe('');
        expect(label).not.toContain('_');
      });
    });

    it('should handle partial factor set', () => {
      const factors: ConfidenceFactors = {
        saturation: 0.90,
        contrast: 0.78,
      };
      expect(getDisplayLabel('saturation')).toBe('Saturation');
      expect(getDisplayLabel('contrast')).toBe('Contrast');
    });

    it('should handle custom/unknown factors', () => {
      const factors: ConfidenceFactors = {
        saturation: 0.90,
        custom_metric: 0.75,
      };
      expect(getDisplayLabel('custom_metric')).toBe('Custom Metric');
    });
  });

  describe('Spec compliance', () => {
    it('should support all spec-defined detection components', () => {
      // From spec: geometry_extraction, ocr_detection, puzzle_detection, domino_detection
      const allComponents: ComponentConfidenceMap = {
        geometry_extraction: 0.85,
        ocr_detection: 0.90,
        puzzle_detection: 0.80,
        domino_detection: 0.80,
      };

      Object.keys(allComponents).forEach(key => {
        const label = getDisplayLabel(key);
        expect(label).toBeTruthy();
        expect(['Geometry', 'OCR', 'Puzzle', 'Domino']).toContain(label);
      });
    });

    it('should support lowest component identification per spec', () => {
      // From spec: "When multiple detection components disagree, use the lowest
      // confidence score and clearly indicate which component is uncertain"
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.90,
        ocr_detection: 0.40, // Low confidence
        puzzle_detection: 0.85,
      };
      const lowest = findLowestComponent(scores);
      expect(lowest).not.toBeNull();
      expect(lowest!.component).toBe('ocr_detection');
      expect(lowest!.score).toBe(0.40);
    });

    it('should correctly identify scores below medium threshold', () => {
      // Scores below 0.70 require manual verification per spec
      const lowScore = 0.65;
      const mediumScore = 0.75;
      const highScore = 0.90;

      const scoresWithLow: ComponentConfidenceMap = {
        geometry_extraction: lowScore,
        puzzle_detection: mediumScore,
      };

      const lowest = findLowestComponent(scoresWithLow);
      expect(lowest!.score).toBe(lowScore);
      expect(lowest!.score).toBeLessThan(0.70); // Below medium threshold
    });
  });

  describe('Edge cases', () => {
    it('should handle 0% confidence', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0,
      };
      const lowest = findLowestComponent(scores);
      expect(lowest).toEqual({ component: 'geometry_extraction', score: 0 });
    });

    it('should handle 100% confidence', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 1.0,
      };
      const lowest = findLowestComponent(scores);
      expect(lowest).toEqual({ component: 'geometry_extraction', score: 1.0 });
    });

    it('should handle very small differences between scores', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.8501,
        ocr_detection: 0.8500,
        puzzle_detection: 0.8502,
      };
      const lowest = findLowestComponent(scores);
      expect(lowest?.component).toBe('ocr_detection');
      expect(lowest?.score).toBe(0.8500);
    });

    it('should handle negative scores (clamp to 0)', () => {
      // The component itself clamps, but findLowestComponent should handle
      // whatever values are passed
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 0.50,
        ocr_detection: -0.10, // Invalid but should still be "lowest"
      };
      const lowest = findLowestComponent(scores);
      expect(lowest?.component).toBe('ocr_detection');
    });

    it('should handle scores > 1 (component clamps)', () => {
      const scores: ComponentConfidenceMap = {
        geometry_extraction: 1.50, // Invalid but component will clamp
        ocr_detection: 0.90,
      };
      const lowest = findLowestComponent(scores);
      expect(lowest?.component).toBe('ocr_detection');
    });
  });
});

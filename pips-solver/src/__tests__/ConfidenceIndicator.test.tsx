/**
 * Tests for ConfidenceIndicator component
 *
 * Verifies that color thresholds match the spec:
 * - high: >= 0.85 (green #10b981)
 * - medium: >= 0.70 (amber #f59e0b)
 * - low: < 0.70 (red #ef4444)
 */

import {
  getConfidenceLevel,
  getConfidenceColor,
  getConfidenceMessage,
  isBorderlineConfidence,
  CONFIDENCE_THRESHOLDS,
  CONFIDENCE_COLORS,
  CONFIDENCE_MESSAGES,
  ConfidenceLevel,
} from '../app/components/ConfidenceIndicator';

describe('ConfidenceIndicator', () => {
  describe('CONFIDENCE_THRESHOLDS', () => {
    it('should have high threshold at 0.85', () => {
      expect(CONFIDENCE_THRESHOLDS.high).toBe(0.85);
    });

    it('should have medium threshold at 0.70', () => {
      expect(CONFIDENCE_THRESHOLDS.medium).toBe(0.70);
    });
  });

  describe('CONFIDENCE_COLORS', () => {
    it('should use green (#10b981) for high confidence', () => {
      expect(CONFIDENCE_COLORS.high).toBe('#10b981');
    });

    it('should use amber (#f59e0b) for medium confidence', () => {
      expect(CONFIDENCE_COLORS.medium).toBe('#f59e0b');
    });

    it('should use red (#ef4444) for low confidence', () => {
      expect(CONFIDENCE_COLORS.low).toBe('#ef4444');
    });
  });

  describe('getConfidenceLevel', () => {
    // High confidence tests (>= 0.85)
    it('should return "high" for confidence >= 0.85', () => {
      expect(getConfidenceLevel(0.85)).toBe('high');
      expect(getConfidenceLevel(0.90)).toBe('high');
      expect(getConfidenceLevel(0.95)).toBe('high');
      expect(getConfidenceLevel(1.0)).toBe('high');
    });

    // Medium confidence tests (>= 0.70 and < 0.85)
    it('should return "medium" for confidence >= 0.70 and < 0.85', () => {
      expect(getConfidenceLevel(0.70)).toBe('medium');
      expect(getConfidenceLevel(0.75)).toBe('medium');
      expect(getConfidenceLevel(0.80)).toBe('medium');
      expect(getConfidenceLevel(0.84)).toBe('medium');
      expect(getConfidenceLevel(0.849)).toBe('medium');
    });

    // Low confidence tests (< 0.70)
    it('should return "low" for confidence < 0.70', () => {
      expect(getConfidenceLevel(0.0)).toBe('low');
      expect(getConfidenceLevel(0.30)).toBe('low');
      expect(getConfidenceLevel(0.50)).toBe('low');
      expect(getConfidenceLevel(0.69)).toBe('low');
      expect(getConfidenceLevel(0.699)).toBe('low');
    });

    // Boundary tests
    it('should handle exact boundary at 0.85 as high', () => {
      expect(getConfidenceLevel(0.85)).toBe('high');
    });

    it('should handle just below 0.85 as medium', () => {
      expect(getConfidenceLevel(0.8499999)).toBe('medium');
    });

    it('should handle exact boundary at 0.70 as medium', () => {
      expect(getConfidenceLevel(0.70)).toBe('medium');
    });

    it('should handle just below 0.70 as low', () => {
      expect(getConfidenceLevel(0.6999999)).toBe('low');
    });
  });

  describe('getConfidenceColor', () => {
    it('should return green for high confidence scores', () => {
      expect(getConfidenceColor(0.90)).toBe('#10b981');
      expect(getConfidenceColor(0.85)).toBe('#10b981');
    });

    it('should return amber for medium confidence scores', () => {
      expect(getConfidenceColor(0.75)).toBe('#f59e0b');
      expect(getConfidenceColor(0.70)).toBe('#f59e0b');
    });

    it('should return red for low confidence scores', () => {
      expect(getConfidenceColor(0.50)).toBe('#ef4444');
      expect(getConfidenceColor(0.69)).toBe('#ef4444');
    });

    it('should accept confidence level strings directly', () => {
      expect(getConfidenceColor('high' as ConfidenceLevel)).toBe('#10b981');
      expect(getConfidenceColor('medium' as ConfidenceLevel)).toBe('#f59e0b');
      expect(getConfidenceColor('low' as ConfidenceLevel)).toBe('#ef4444');
    });
  });

  describe('getConfidenceMessage', () => {
    it('should return appropriate message for high confidence', () => {
      expect(getConfidenceMessage(0.90)).toBe('High confidence - likely accurate');
      expect(getConfidenceMessage('high' as ConfidenceLevel)).toBe('High confidence - likely accurate');
    });

    it('should return appropriate message for medium confidence', () => {
      expect(getConfidenceMessage(0.75)).toBe('Medium confidence - review recommended');
      expect(getConfidenceMessage('medium' as ConfidenceLevel)).toBe('Medium confidence - review recommended');
    });

    it('should return appropriate message for low confidence', () => {
      expect(getConfidenceMessage(0.50)).toBe('Low confidence - manual verification required');
      expect(getConfidenceMessage('low' as ConfidenceLevel)).toBe('Low confidence - manual verification required');
    });
  });

  describe('isBorderlineConfidence', () => {
    // Near high threshold (0.85)
    it('should detect borderline near high threshold', () => {
      expect(isBorderlineConfidence(0.82)).toBe(true);  // 0.85 - 0.03
      expect(isBorderlineConfidence(0.88)).toBe(true);  // 0.85 + 0.03
      expect(isBorderlineConfidence(0.81)).toBe(true);  // 0.85 - 0.04
      expect(isBorderlineConfidence(0.89)).toBe(true);  // 0.85 + 0.04
    });

    // Near medium threshold (0.70)
    it('should detect borderline near medium threshold', () => {
      expect(isBorderlineConfidence(0.67)).toBe(true);  // 0.70 - 0.03
      expect(isBorderlineConfidence(0.73)).toBe(true);  // 0.70 + 0.03
      expect(isBorderlineConfidence(0.66)).toBe(true);  // 0.70 - 0.04
      expect(isBorderlineConfidence(0.74)).toBe(true);  // 0.70 + 0.04
    });

    // Not borderline
    it('should not detect borderline when far from thresholds', () => {
      expect(isBorderlineConfidence(0.50)).toBe(false);  // Far from both
      expect(isBorderlineConfidence(0.95)).toBe(false);  // Far from both
      expect(isBorderlineConfidence(0.60)).toBe(false);  // 0.10 from medium
    });
  });

  describe('Spec compliance', () => {
    it('should match spec high threshold of 0.85', () => {
      // From spec: high >= 0.85 should be green
      expect(getConfidenceLevel(0.85)).toBe('high');
      expect(getConfidenceColor(0.85)).toBe('#10b981');
    });

    it('should match spec medium threshold of 0.70', () => {
      // From spec: medium >= 0.70 should be amber
      expect(getConfidenceLevel(0.70)).toBe('medium');
      expect(getConfidenceColor(0.70)).toBe('#f59e0b');
    });

    it('should match spec low threshold (below 0.70)', () => {
      // From spec: low < 0.70 should be red
      expect(getConfidenceLevel(0.69)).toBe('low');
      expect(getConfidenceColor(0.69)).toBe('#ef4444');
    });
  });
});

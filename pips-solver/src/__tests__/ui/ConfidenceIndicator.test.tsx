/**
 * Tests for Themed ConfidenceIndicator UI component
 *
 * Verifies that color thresholds match the spec:
 * - high: >= 0.85 (jade #00A878)
 * - medium: >= 0.70 (brass #C9A227)
 * - low: < 0.70 (coral #FF6B6B)
 */

import {
  getConfidenceLevel,
  getConfidenceColor,
  getConfidenceClassName,
  getConfidenceMessage,
  isBorderlineConfidence,
  CONFIDENCE_THRESHOLDS,
  CONFIDENCE_COLORS,
  CONFIDENCE_CLASS_NAMES,
  CONFIDENCE_MESSAGES,
  ConfidenceLevel,
} from '../../app/components/ui/ConfidenceIndicator';

describe('Themed ConfidenceIndicator (ui)', () => {
  describe('CONFIDENCE_THRESHOLDS', () => {
    it('should have high threshold at 0.85', () => {
      expect(CONFIDENCE_THRESHOLDS.high).toBe(0.85);
    });

    it('should have medium threshold at 0.70', () => {
      expect(CONFIDENCE_THRESHOLDS.medium).toBe(0.70);
    });
  });

  describe('CONFIDENCE_COLORS (jade/brass/coral theme)', () => {
    it('should use jade (#00A878) for high confidence', () => {
      expect(CONFIDENCE_COLORS.high).toBe('#00A878');
    });

    it('should use brass (#C9A227) for medium confidence', () => {
      expect(CONFIDENCE_COLORS.medium).toBe('#C9A227');
    });

    it('should use coral (#FF6B6B) for low confidence', () => {
      expect(CONFIDENCE_COLORS.low).toBe('#FF6B6B');
    });
  });

  describe('CONFIDENCE_CLASS_NAMES', () => {
    it('should have confidence-jade class for high', () => {
      expect(CONFIDENCE_CLASS_NAMES.high).toBe('confidence-jade');
    });

    it('should have confidence-brass class for medium', () => {
      expect(CONFIDENCE_CLASS_NAMES.medium).toBe('confidence-brass');
    });

    it('should have confidence-coral class for low', () => {
      expect(CONFIDENCE_CLASS_NAMES.low).toBe('confidence-coral');
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

  describe('getConfidenceColor (themed)', () => {
    it('should return jade for high confidence scores', () => {
      expect(getConfidenceColor(0.90)).toBe('#00A878');
      expect(getConfidenceColor(0.85)).toBe('#00A878');
    });

    it('should return brass for medium confidence scores', () => {
      expect(getConfidenceColor(0.75)).toBe('#C9A227');
      expect(getConfidenceColor(0.70)).toBe('#C9A227');
    });

    it('should return coral for low confidence scores', () => {
      expect(getConfidenceColor(0.50)).toBe('#FF6B6B');
      expect(getConfidenceColor(0.69)).toBe('#FF6B6B');
    });

    it('should accept confidence level strings directly', () => {
      expect(getConfidenceColor('high' as ConfidenceLevel)).toBe('#00A878');
      expect(getConfidenceColor('medium' as ConfidenceLevel)).toBe('#C9A227');
      expect(getConfidenceColor('low' as ConfidenceLevel)).toBe('#FF6B6B');
    });
  });

  describe('getConfidenceClassName', () => {
    it('should return jade class for high confidence scores', () => {
      expect(getConfidenceClassName(0.90)).toBe('confidence-jade');
      expect(getConfidenceClassName(0.85)).toBe('confidence-jade');
    });

    it('should return brass class for medium confidence scores', () => {
      expect(getConfidenceClassName(0.75)).toBe('confidence-brass');
      expect(getConfidenceClassName(0.70)).toBe('confidence-brass');
    });

    it('should return coral class for low confidence scores', () => {
      expect(getConfidenceClassName(0.50)).toBe('confidence-coral');
      expect(getConfidenceClassName(0.69)).toBe('confidence-coral');
    });

    it('should accept confidence level strings directly', () => {
      expect(getConfidenceClassName('high' as ConfidenceLevel)).toBe('confidence-jade');
      expect(getConfidenceClassName('medium' as ConfidenceLevel)).toBe('confidence-brass');
      expect(getConfidenceClassName('low' as ConfidenceLevel)).toBe('confidence-coral');
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

  describe('Spec compliance (jade/brass/coral theme)', () => {
    it('should match spec high threshold of 0.85 with jade color', () => {
      // From spec: high >= 0.85 should be jade (green)
      expect(getConfidenceLevel(0.85)).toBe('high');
      expect(getConfidenceColor(0.85)).toBe('#00A878');
      expect(getConfidenceClassName(0.85)).toBe('confidence-jade');
    });

    it('should match spec medium threshold of 0.70 with brass color', () => {
      // From spec: medium >= 0.70 should be brass (amber/gold)
      expect(getConfidenceLevel(0.70)).toBe('medium');
      expect(getConfidenceColor(0.70)).toBe('#C9A227');
      expect(getConfidenceClassName(0.70)).toBe('confidence-brass');
    });

    it('should match spec low threshold (below 0.70) with coral color', () => {
      // From spec: low < 0.70 should be coral (red)
      expect(getConfidenceLevel(0.69)).toBe('low');
      expect(getConfidenceColor(0.69)).toBe('#FF6B6B');
      expect(getConfidenceClassName(0.69)).toBe('confidence-coral');
    });
  });
});

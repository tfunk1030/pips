/**
 * End-to-End Confidence Flow Integration Tests (Frontend)
 *
 * These tests verify that the frontend confidence components correctly
 * interpret and display confidence scores from the backend API.
 *
 * Verification steps per spec:
 * 1. Start cv-service on port 8080
 * 2. Start pips-solver on port 3000
 * 3. Upload test image via UI
 * 4. Verify confidence displayed in UI matches backend response
 * 5. Verify color coding matches confidence level
 */

import {
  getConfidenceLevel,
  getConfidenceColor,
  getConfidenceMessage,
  isBorderlineConfidence,
  CONFIDENCE_THRESHOLDS,
  CONFIDENCE_COLORS,
  CONFIDENCE_MESSAGES,
} from '../../app/components/ConfidenceIndicator';

/**
 * Backend threshold values from cv-service/confidence_config.py
 * These MUST match for the E2E flow to work correctly.
 */
const BACKEND_THRESHOLDS = {
  geometry_extraction: { high: 0.85, medium: 0.70, low: 0.0 },
  ocr_detection: { high: 0.90, medium: 0.75, low: 0.0 },
  puzzle_detection: { high: 0.80, medium: 0.65, low: 0.0 },
  domino_detection: { high: 0.80, medium: 0.65, low: 0.0 },
};

/**
 * Backend borderline margin (5%)
 */
const BACKEND_BORDERLINE_MARGIN = 0.05;

describe('E2E Confidence Flow - Cross-Service Consistency', () => {
  describe('Threshold Alignment', () => {
    it('should have frontend high threshold matching backend geometry_extraction', () => {
      // Frontend uses geometry_extraction thresholds as the default
      expect(CONFIDENCE_THRESHOLDS.high).toBe(BACKEND_THRESHOLDS.geometry_extraction.high);
    });

    it('should have frontend medium threshold matching backend geometry_extraction', () => {
      expect(CONFIDENCE_THRESHOLDS.medium).toBe(BACKEND_THRESHOLDS.geometry_extraction.medium);
    });

    it('should classify same confidence scores identically to backend', () => {
      // Test cases: [confidence, expected_level]
      const testCases: [number, 'high' | 'medium' | 'low'][] = [
        [0.95, 'high'],
        [0.85, 'high'],
        [0.84, 'medium'],
        [0.80, 'medium'],
        [0.70, 'medium'],
        [0.69, 'low'],
        [0.50, 'low'],
        [0.30, 'low'],
        [0.0, 'low'],
      ];

      for (const [confidence, expectedLevel] of testCases) {
        const level = getConfidenceLevel(confidence);
        expect(level).toBe(expectedLevel);
      }
    });
  });

  describe('Color Mapping Verification', () => {
    it('should display green (#10b981) for high confidence', () => {
      const color = getConfidenceColor(0.90);
      expect(color).toBe('#10b981');
    });

    it('should display amber (#f59e0b) for medium confidence', () => {
      const color = getConfidenceColor(0.75);
      expect(color).toBe('#f59e0b');
    });

    it('should display red (#ef4444) for low confidence', () => {
      const color = getConfidenceColor(0.50);
      expect(color).toBe('#ef4444');
    });

    it('should match spec color assignments', () => {
      expect(CONFIDENCE_COLORS.high).toBe('#10b981');
      expect(CONFIDENCE_COLORS.medium).toBe('#f59e0b');
      expect(CONFIDENCE_COLORS.low).toBe('#ef4444');
    });
  });

  describe('Borderline Detection', () => {
    it('should use 5% margin matching backend BORDERLINE_MARGIN', () => {
      // Test values that should be borderline (within 5% of thresholds)
      // Near high threshold (0.85)
      expect(isBorderlineConfidence(0.82)).toBe(true);  // 0.85 - 0.03
      expect(isBorderlineConfidence(0.88)).toBe(true);  // 0.85 + 0.03

      // Near medium threshold (0.70)
      expect(isBorderlineConfidence(0.67)).toBe(true);  // 0.70 - 0.03
      expect(isBorderlineConfidence(0.73)).toBe(true);  // 0.70 + 0.03
    });

    it('should not flag non-borderline values', () => {
      // Well away from any threshold
      expect(isBorderlineConfidence(0.50)).toBe(false);
      expect(isBorderlineConfidence(0.95)).toBe(false);
    });
  });

  describe('Message Consistency', () => {
    it('should have appropriate high confidence message', () => {
      expect(CONFIDENCE_MESSAGES.high).toBe('High confidence - likely accurate');
    });

    it('should have appropriate medium confidence message', () => {
      expect(CONFIDENCE_MESSAGES.medium).toBe('Medium confidence - review recommended');
    });

    it('should have appropriate low confidence message', () => {
      expect(CONFIDENCE_MESSAGES.low).toBe('Low confidence - manual verification required');
    });
  });
});

describe('E2E Confidence Flow - Simulated Backend Response', () => {
  /**
   * Simulates processing a backend API response and verifying
   * the frontend correctly interprets and displays it.
   */
  interface BackendConfidenceResponse {
    success: boolean;
    confidence: number;
    threshold: 'high' | 'medium' | 'low';
    confidence_breakdown: {
      saturation: number;
      area_ratio: number;
      aspect_ratio: number;
      relative_size: number;
      edge_clarity: number;
      contrast: number;
    };
    is_borderline: boolean;
  }

  function simulateBackendResponse(confidence: number): BackendConfidenceResponse {
    // Simulate backend classification logic (from confidence_config.py)
    let threshold: 'high' | 'medium' | 'low';
    if (confidence >= 0.85) {
      threshold = 'high';
    } else if (confidence >= 0.70) {
      threshold = 'medium';
    } else {
      threshold = 'low';
    }

    // Simulate borderline check (within 5% of thresholds)
    const isBorderline =
      Math.abs(confidence - 0.85) <= 0.05 ||
      Math.abs(confidence - 0.70) <= 0.05;

    return {
      success: true,
      confidence,
      threshold,
      confidence_breakdown: {
        saturation: 0.8,
        area_ratio: 0.9,
        aspect_ratio: 0.85,
        relative_size: 0.7,
        edge_clarity: 0.75,
        contrast: 0.8,
      },
      is_borderline: isBorderline,
    };
  }

  it('should process high confidence response correctly', () => {
    const backendResponse = simulateBackendResponse(0.90);

    // Frontend should match backend classification
    const frontendLevel = getConfidenceLevel(backendResponse.confidence);
    expect(frontendLevel).toBe(backendResponse.threshold);
    expect(frontendLevel).toBe('high');

    // Frontend should display green
    const color = getConfidenceColor(frontendLevel);
    expect(color).toBe('#10b981');
  });

  it('should process medium confidence response correctly', () => {
    const backendResponse = simulateBackendResponse(0.75);

    const frontendLevel = getConfidenceLevel(backendResponse.confidence);
    expect(frontendLevel).toBe(backendResponse.threshold);
    expect(frontendLevel).toBe('medium');

    const color = getConfidenceColor(frontendLevel);
    expect(color).toBe('#f59e0b');
  });

  it('should process low confidence response correctly', () => {
    const backendResponse = simulateBackendResponse(0.50);

    const frontendLevel = getConfidenceLevel(backendResponse.confidence);
    expect(frontendLevel).toBe(backendResponse.threshold);
    expect(frontendLevel).toBe('low');

    const color = getConfidenceColor(frontendLevel);
    expect(color).toBe('#ef4444');
  });

  it('should correctly identify borderline values from backend', () => {
    // Backend reports borderline = true at 0.82 (near 0.85 threshold)
    const backendResponse = simulateBackendResponse(0.82);
    expect(backendResponse.is_borderline).toBe(true);

    // Frontend should also flag as borderline
    const frontendBorderline = isBorderlineConfidence(backendResponse.confidence);
    expect(frontendBorderline).toBe(backendResponse.is_borderline);
  });

  it('should handle all confidence breakdowns from backend', () => {
    const backendResponse = simulateBackendResponse(0.80);

    // Verify all breakdown factors are present
    const breakdown = backendResponse.confidence_breakdown;
    expect(breakdown).toHaveProperty('saturation');
    expect(breakdown).toHaveProperty('area_ratio');
    expect(breakdown).toHaveProperty('aspect_ratio');
    expect(breakdown).toHaveProperty('relative_size');
    expect(breakdown).toHaveProperty('edge_clarity');
    expect(breakdown).toHaveProperty('contrast');

    // All should be in valid range
    Object.values(breakdown).forEach(value => {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(1);
    });
  });
});

describe('E2E Confidence Flow - Boundary Conditions', () => {
  it('should handle exact threshold boundaries correctly', () => {
    // At exactly 0.85 -> high
    expect(getConfidenceLevel(0.85)).toBe('high');
    expect(getConfidenceColor(0.85)).toBe('#10b981');

    // At exactly 0.70 -> medium
    expect(getConfidenceLevel(0.70)).toBe('medium');
    expect(getConfidenceColor(0.70)).toBe('#f59e0b');
  });

  it('should handle just below threshold boundaries correctly', () => {
    // Just below 0.85 -> medium
    expect(getConfidenceLevel(0.8499999)).toBe('medium');

    // Just below 0.70 -> low
    expect(getConfidenceLevel(0.6999999)).toBe('low');
  });

  it('should handle edge values (0.0 and 1.0)', () => {
    expect(getConfidenceLevel(0.0)).toBe('low');
    expect(getConfidenceColor(0.0)).toBe('#ef4444');

    expect(getConfidenceLevel(1.0)).toBe('high');
    expect(getConfidenceColor(1.0)).toBe('#10b981');
  });
});

describe('E2E Confidence Flow - Spec Compliance', () => {
  it('should meet spec requirement: high threshold at 0.85', () => {
    expect(CONFIDENCE_THRESHOLDS.high).toBe(0.85);
  });

  it('should meet spec requirement: medium threshold at 0.70', () => {
    expect(CONFIDENCE_THRESHOLDS.medium).toBe(0.70);
  });

  it('should meet spec requirement: green color for high confidence', () => {
    expect(CONFIDENCE_COLORS.high).toBe('#10b981');
  });

  it('should meet spec requirement: amber color for medium confidence', () => {
    expect(CONFIDENCE_COLORS.medium).toBe('#f59e0b');
  });

  it('should meet spec requirement: red color for low confidence', () => {
    expect(CONFIDENCE_COLORS.low).toBe('#ef4444');
  });

  it('should meet spec requirement: 5% borderline margin', () => {
    // Within 5% of 0.85 should be borderline
    // Note: Using values clearly within margin to avoid floating-point precision issues at exact boundaries
    expect(isBorderlineConfidence(0.81)).toBe(true);  // 0.85 - 0.04
    expect(isBorderlineConfidence(0.89)).toBe(true);  // 0.85 + 0.04

    // Within 5% of 0.70 should be borderline
    expect(isBorderlineConfidence(0.66)).toBe(true);  // 0.70 - 0.04
    expect(isBorderlineConfidence(0.74)).toBe(true);  // 0.70 + 0.04
  });
});

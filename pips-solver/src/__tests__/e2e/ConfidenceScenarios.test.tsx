/**
 * High and Low Confidence Scenario Tests (Frontend)
 *
 * This test module validates the UI components correctly display confidence
 * indicators for high and low confidence scenarios.
 *
 * Verification per subtask-4-2:
 * - Test with clear image (expect high confidence, green)
 * - Test with blurry image (expect low confidence, red/amber)
 * - Verify UI messages match confidence level
 */

import React from 'react';
import { render, screen } from '@testing-library/react';

import {
  ConfidenceIndicator,
  getConfidenceLevel,
  getConfidenceColor,
  getConfidenceMessage,
  isBorderlineConfidence,
  CONFIDENCE_THRESHOLDS,
  CONFIDENCE_COLORS,
  CONFIDENCE_MESSAGES,
} from '../../app/components/ConfidenceIndicator';

// ============================================================================
// High Confidence Scenarios
// ============================================================================

describe('High Confidence Scenarios', () => {
  /**
   * Scenario: Clear, high-quality image detection
   * Expected: Green indicator, "High confidence - likely accurate" message
   */

  describe('Visual Indicator', () => {
    it('should display green color for high confidence (>= 0.85)', () => {
      const highConfidenceValues = [0.85, 0.90, 0.95, 0.99, 1.0];

      for (const confidence of highConfidenceValues) {
        const level = getConfidenceLevel(confidence);
        const color = getConfidenceColor(confidence);

        expect(level).toBe('high');
        expect(color).toBe('#10b981'); // Green
      }
    });

    it('should render green indicator component for clear image confidence', () => {
      render(<ConfidenceIndicator confidence={0.92} />);

      const indicator = screen.getByTestId('confidence-indicator');
      const percentage = screen.getByTestId('confidence-percentage');

      expect(indicator).toBeInTheDocument();
      expect(percentage).toHaveTextContent('92%');
      expect(percentage).toHaveStyle({ color: '#10b981' });
    });

    it('should show checkmark icon for high confidence', () => {
      render(<ConfidenceIndicator confidence={0.90} />);

      const icon = screen.getByTestId('confidence-icon');
      expect(icon).toHaveTextContent('\u2713'); // checkmark
    });
  });

  describe('Message Display', () => {
    it('should display "High confidence - likely accurate" message', () => {
      render(<ConfidenceIndicator confidence={0.90} showMessage={true} />);

      const message = screen.getByTestId('confidence-message');
      expect(message).toHaveTextContent('High confidence - likely accurate');
    });

    it('should have correct message in CONFIDENCE_MESSAGES', () => {
      expect(CONFIDENCE_MESSAGES.high).toBe('High confidence - likely accurate');
    });

    it('should return correct message from getConfidenceMessage', () => {
      expect(getConfidenceMessage(0.90)).toBe('High confidence - likely accurate');
      expect(getConfidenceMessage('high')).toBe('High confidence - likely accurate');
    });
  });

  describe('Component Display', () => {
    it('should display component name when provided', () => {
      render(<ConfidenceIndicator confidence={0.90} component="geometry" />);

      const component = screen.getByTestId('confidence-component');
      expect(component).toHaveTextContent('geometry');
    });
  });
});

// ============================================================================
// Low Confidence Scenarios
// ============================================================================

describe('Low Confidence Scenarios', () => {
  /**
   * Scenario: Blurry, low-quality image detection
   * Expected: Red indicator, "Low confidence - manual verification required" message
   */

  describe('Visual Indicator', () => {
    it('should display red color for low confidence (< 0.70)', () => {
      const lowConfidenceValues = [0.0, 0.10, 0.30, 0.50, 0.69];

      for (const confidence of lowConfidenceValues) {
        const level = getConfidenceLevel(confidence);
        const color = getConfidenceColor(confidence);

        expect(level).toBe('low');
        expect(color).toBe('#ef4444'); // Red
      }
    });

    it('should render red indicator component for blurry image confidence', () => {
      render(<ConfidenceIndicator confidence={0.45} />);

      const percentage = screen.getByTestId('confidence-percentage');

      expect(percentage).toHaveTextContent('45%');
      expect(percentage).toHaveStyle({ color: '#ef4444' });
    });

    it('should show warning icon for low confidence', () => {
      render(<ConfidenceIndicator confidence={0.40} />);

      const icon = screen.getByTestId('confidence-icon');
      expect(icon).toHaveTextContent('\u26A0'); // warning triangle
    });
  });

  describe('Message Display', () => {
    it('should display "Low confidence - manual verification required" message', () => {
      render(<ConfidenceIndicator confidence={0.40} showMessage={true} />);

      const message = screen.getByTestId('confidence-message');
      expect(message).toHaveTextContent('Low confidence - manual verification required');
    });

    it('should have correct message in CONFIDENCE_MESSAGES', () => {
      expect(CONFIDENCE_MESSAGES.low).toBe('Low confidence - manual verification required');
    });

    it('should return correct message from getConfidenceMessage', () => {
      expect(getConfidenceMessage(0.40)).toBe('Low confidence - manual verification required');
      expect(getConfidenceMessage('low')).toBe('Low confidence - manual verification required');
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero confidence', () => {
      render(<ConfidenceIndicator confidence={0} />);

      const percentage = screen.getByTestId('confidence-percentage');
      expect(percentage).toHaveTextContent('0%');
      expect(percentage).toHaveStyle({ color: '#ef4444' });
    });

    it('should handle negative confidence (clamp to 0)', () => {
      render(<ConfidenceIndicator confidence={-0.5} />);

      const percentage = screen.getByTestId('confidence-percentage');
      expect(percentage).toHaveTextContent('0%');
    });
  });
});

// ============================================================================
// Medium (Borderline) Confidence Scenarios
// ============================================================================

describe('Medium Confidence Scenarios', () => {
  /**
   * Scenario: Medium quality image detection
   * Expected: Amber indicator, "Medium confidence - review recommended" message
   */

  describe('Visual Indicator', () => {
    it('should display amber color for medium confidence (0.70 - 0.85)', () => {
      const mediumConfidenceValues = [0.70, 0.75, 0.80, 0.84];

      for (const confidence of mediumConfidenceValues) {
        const level = getConfidenceLevel(confidence);
        const color = getConfidenceColor(confidence);

        expect(level).toBe('medium');
        expect(color).toBe('#f59e0b'); // Amber
      }
    });

    it('should render amber indicator for medium quality detection', () => {
      render(<ConfidenceIndicator confidence={0.78} />);

      const percentage = screen.getByTestId('confidence-percentage');

      expect(percentage).toHaveTextContent('78%');
      expect(percentage).toHaveStyle({ color: '#f59e0b' });
    });
  });

  describe('Message Display', () => {
    it('should display "Medium confidence - review recommended" message', () => {
      render(<ConfidenceIndicator confidence={0.75} showMessage={true} />);

      const message = screen.getByTestId('confidence-message');
      expect(message).toHaveTextContent('Medium confidence - review recommended');
    });
  });

  describe('Borderline Detection', () => {
    it('should show borderline indicator near high threshold', () => {
      // 0.82 is within 5% of 0.85
      render(<ConfidenceIndicator confidence={0.82} />);

      const borderline = screen.getByTestId('confidence-borderline');
      expect(borderline).toHaveTextContent('(borderline - near threshold)');
    });

    it('should show borderline indicator near medium threshold', () => {
      // 0.72 is within 5% of 0.70
      render(<ConfidenceIndicator confidence={0.72} />);

      const borderline = screen.getByTestId('confidence-borderline');
      expect(borderline).toHaveTextContent('(borderline - near threshold)');
    });

    it('should NOT show borderline for values far from thresholds', () => {
      render(<ConfidenceIndicator confidence={0.50} />);

      const borderline = screen.queryByTestId('confidence-borderline');
      expect(borderline).not.toBeInTheDocument();
    });
  });
});

// ============================================================================
// Threshold Boundary Tests
// ============================================================================

describe('Threshold Boundary Scenarios', () => {
  it('should classify exactly 0.85 as high', () => {
    const level = getConfidenceLevel(0.85);
    expect(level).toBe('high');
  });

  it('should classify 0.8499999 as medium', () => {
    const level = getConfidenceLevel(0.8499999);
    expect(level).toBe('medium');
  });

  it('should classify exactly 0.70 as medium', () => {
    const level = getConfidenceLevel(0.70);
    expect(level).toBe('medium');
  });

  it('should classify 0.6999999 as low', () => {
    const level = getConfidenceLevel(0.6999999);
    expect(level).toBe('low');
  });

  it('should classify exactly 1.0 as high', () => {
    const level = getConfidenceLevel(1.0);
    expect(level).toBe('high');
  });

  it('should classify exactly 0.0 as low', () => {
    const level = getConfidenceLevel(0.0);
    expect(level).toBe('low');
  });
});

// ============================================================================
// UI Message Matching Verification
// ============================================================================

describe('UI Message Matching Verification', () => {
  /**
   * Per spec, UI messages must match confidence levels:
   * - High: "High confidence - likely accurate"
   * - Medium: "Medium confidence - review recommended"
   * - Low: "Low confidence - manual verification required"
   */

  const scenarios = [
    {
      name: 'Clear Image (High Confidence)',
      confidence: 0.92,
      expectedLevel: 'high',
      expectedColor: '#10b981',
      expectedMessage: 'High confidence - likely accurate',
      expectedColorName: 'Green',
    },
    {
      name: 'Medium Quality Image (Medium Confidence)',
      confidence: 0.78,
      expectedLevel: 'medium',
      expectedColor: '#f59e0b',
      expectedMessage: 'Medium confidence - review recommended',
      expectedColorName: 'Amber',
    },
    {
      name: 'Blurry Image (Low Confidence)',
      confidence: 0.45,
      expectedLevel: 'low',
      expectedColor: '#ef4444',
      expectedMessage: 'Low confidence - manual verification required',
      expectedColorName: 'Red',
    },
  ];

  scenarios.forEach(scenario => {
    describe(scenario.name, () => {
      it(`should have level = ${scenario.expectedLevel}`, () => {
        expect(getConfidenceLevel(scenario.confidence)).toBe(scenario.expectedLevel);
      });

      it(`should have color = ${scenario.expectedColorName} (${scenario.expectedColor})`, () => {
        expect(getConfidenceColor(scenario.confidence)).toBe(scenario.expectedColor);
      });

      it(`should have message = "${scenario.expectedMessage}"`, () => {
        expect(getConfidenceMessage(scenario.confidence)).toBe(scenario.expectedMessage);
      });

      it('should render correctly in ConfidenceIndicator component', () => {
        render(
          <ConfidenceIndicator
            confidence={scenario.confidence}
            showPercentage={true}
            showMessage={true}
          />
        );

        const percentage = screen.getByTestId('confidence-percentage');
        const message = screen.getByTestId('confidence-message');

        expect(percentage).toHaveStyle({ color: scenario.expectedColor });
        expect(message).toHaveTextContent(scenario.expectedMessage);
      });
    });
  });
});

// ============================================================================
// Complete Scenario Flow Tests
// ============================================================================

describe('Complete Scenario Flow', () => {
  it('should correctly process clear image scenario (high confidence)', () => {
    // Simulated backend response for clear image
    const backendResponse = {
      success: true,
      confidence: 0.92,
      threshold: 'high' as const,
      confidence_breakdown: {
        saturation: 0.95,
        area_ratio: 0.88,
        aspect_ratio: 0.92,
        relative_size: 0.85,
        edge_clarity: 0.90,
        contrast: 0.87,
      },
      is_borderline: false,
    };

    // Frontend should display correctly
    const frontendLevel = getConfidenceLevel(backendResponse.confidence);
    const frontendColor = getConfidenceColor(backendResponse.confidence);
    const frontendMessage = getConfidenceMessage(backendResponse.confidence);
    const frontendBorderline = isBorderlineConfidence(backendResponse.confidence);

    // Verify match
    expect(frontendLevel).toBe(backendResponse.threshold);
    expect(frontendLevel).toBe('high');
    expect(frontendColor).toBe('#10b981');
    expect(frontendMessage).toBe('High confidence - likely accurate');
    expect(frontendBorderline).toBe(backendResponse.is_borderline);
  });

  it('should correctly process blurry image scenario (low confidence)', () => {
    // Simulated backend response for blurry image
    const backendResponse = {
      success: true,
      confidence: 0.35,
      threshold: 'low' as const,
      confidence_breakdown: {
        saturation: 0.25,
        area_ratio: 0.40,
        aspect_ratio: 0.65,
        relative_size: 0.30,
        edge_clarity: 0.20,
        contrast: 0.30,
      },
      is_borderline: false,
    };

    const frontendLevel = getConfidenceLevel(backendResponse.confidence);
    const frontendColor = getConfidenceColor(backendResponse.confidence);
    const frontendMessage = getConfidenceMessage(backendResponse.confidence);

    expect(frontendLevel).toBe(backendResponse.threshold);
    expect(frontendLevel).toBe('low');
    expect(frontendColor).toBe('#ef4444');
    expect(frontendMessage).toBe('Low confidence - manual verification required');
  });

  it('should correctly process borderline scenario', () => {
    // Simulated backend response for borderline confidence
    const backendResponse = {
      success: true,
      confidence: 0.82,
      threshold: 'medium' as const,
      is_borderline: true,
    };

    const frontendLevel = getConfidenceLevel(backendResponse.confidence);
    const frontendBorderline = isBorderlineConfidence(backendResponse.confidence);

    expect(frontendLevel).toBe(backendResponse.threshold);
    expect(frontendBorderline).toBe(backendResponse.is_borderline);
  });
});

// ============================================================================
// Spec Compliance Verification
// ============================================================================

describe('Spec Compliance Verification', () => {
  it('should have high threshold at 0.85 per spec', () => {
    expect(CONFIDENCE_THRESHOLDS.high).toBe(0.85);
  });

  it('should have medium threshold at 0.70 per spec', () => {
    expect(CONFIDENCE_THRESHOLDS.medium).toBe(0.70);
  });

  it('should have green (#10b981) for high confidence per spec', () => {
    expect(CONFIDENCE_COLORS.high).toBe('#10b981');
  });

  it('should have amber (#f59e0b) for medium confidence per spec', () => {
    expect(CONFIDENCE_COLORS.medium).toBe('#f59e0b');
  });

  it('should have red (#ef4444) for low confidence per spec', () => {
    expect(CONFIDENCE_COLORS.low).toBe('#ef4444');
  });

  it('should use 5% borderline margin per spec', () => {
    // Within 5% of 0.85 should be borderline
    expect(isBorderlineConfidence(0.81)).toBe(true);  // 0.85 - 0.04
    expect(isBorderlineConfidence(0.89)).toBe(true);  // 0.85 + 0.04

    // Within 5% of 0.70 should be borderline
    expect(isBorderlineConfidence(0.66)).toBe(true);  // 0.70 - 0.04
    expect(isBorderlineConfidence(0.74)).toBe(true);  // 0.70 + 0.04

    // Outside 5% should NOT be borderline
    expect(isBorderlineConfidence(0.50)).toBe(false);
    expect(isBorderlineConfidence(0.95)).toBe(false);
    expect(isBorderlineConfidence(0.77)).toBe(false); // Between thresholds but not near
  });
});

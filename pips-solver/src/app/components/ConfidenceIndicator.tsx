import React from 'react';

// Confidence threshold configuration aligned with cv-service confidence_config.py
// These thresholds determine visual indicators for detection confidence
const CONFIDENCE_THRESHOLDS = {
  high: 0.85,   // User can trust without review
  medium: 0.70, // Suggest review
  // Below 0.70 is considered low confidence
};

// Color palette for confidence levels
const CONFIDENCE_COLORS = {
  high: '#10b981',    // Green - trustworthy
  medium: '#f59e0b',  // Amber - review suggested
  low: '#ef4444',     // Red - verification required
};

// User-facing messages for each confidence level
const CONFIDENCE_MESSAGES = {
  high: 'High confidence - likely accurate',
  medium: 'Medium confidence - review recommended',
  low: 'Low confidence - manual verification required',
};

export type ConfidenceLevel = 'high' | 'medium' | 'low';

export interface ConfidenceIndicatorProps {
  /** Confidence score from 0.0 to 1.0 */
  confidence: number;
  /** Detection component name (e.g., 'geometry', 'ocr', 'puzzle') */
  component?: string;
  /** Whether to show the percentage value */
  showPercentage?: boolean;
  /** Whether to show the confidence message */
  showMessage?: boolean;
  /** Optional custom style */
  style?: React.CSSProperties;
}

/**
 * Determines the confidence level based on the score and thresholds.
 * Uses thresholds: high >= 0.85, medium >= 0.70, low < 0.70
 */
export function getConfidenceLevel(confidence: number): ConfidenceLevel {
  if (confidence >= CONFIDENCE_THRESHOLDS.high) {
    return 'high';
  } else if (confidence >= CONFIDENCE_THRESHOLDS.medium) {
    return 'medium';
  } else {
    return 'low';
  }
}

/**
 * Gets the color for a given confidence level or score.
 */
export function getConfidenceColor(confidenceOrLevel: number | ConfidenceLevel): string {
  const level = typeof confidenceOrLevel === 'number'
    ? getConfidenceLevel(confidenceOrLevel)
    : confidenceOrLevel;
  return CONFIDENCE_COLORS[level];
}

/**
 * Gets the message for a given confidence level or score.
 */
export function getConfidenceMessage(confidenceOrLevel: number | ConfidenceLevel): string {
  const level = typeof confidenceOrLevel === 'number'
    ? getConfidenceLevel(confidenceOrLevel)
    : confidenceOrLevel;
  return CONFIDENCE_MESSAGES[level];
}

/**
 * Checks if confidence is within 5% of a threshold boundary (borderline).
 * Useful for adding visual indicators that score is near a threshold.
 */
export function isBorderlineConfidence(confidence: number): boolean {
  const distanceToHighThreshold = Math.abs(confidence - CONFIDENCE_THRESHOLDS.high);
  const distanceToMediumThreshold = Math.abs(confidence - CONFIDENCE_THRESHOLDS.medium);
  return distanceToHighThreshold <= 0.05 || distanceToMediumThreshold <= 0.05;
}

/**
 * Returns the appropriate icon for the confidence level.
 */
function getConfidenceIcon(level: ConfidenceLevel): string {
  switch (level) {
    case 'high':
      return '\u2713'; // checkmark
    case 'medium':
      return '\u26A0'; // warning triangle
    case 'low':
      return '\u26A0'; // warning triangle
    default:
      return '';
  }
}

/**
 * ConfidenceIndicator - Displays visual confidence indicators for detection results.
 *
 * Color thresholds match the spec requirements:
 * - high: >= 0.85 (green #10b981) - User can trust without review
 * - medium: >= 0.70 (amber #f59e0b) - Suggest review
 * - low: < 0.70 (red #ef4444) - Requires manual verification
 */
export const ConfidenceIndicator: React.FC<ConfidenceIndicatorProps> = ({
  confidence,
  component,
  showPercentage = true,
  showMessage = true,
  style,
}) => {
  // Clamp confidence to valid range
  const normalizedConfidence = Math.max(0, Math.min(1, confidence));
  const level = getConfidenceLevel(normalizedConfidence);
  const color = getConfidenceColor(level);
  const message = getConfidenceMessage(level);
  const icon = getConfidenceIcon(level);
  const isBorderline = isBorderlineConfidence(normalizedConfidence);
  const percentage = Math.round(normalizedConfidence * 100);

  const containerStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    ...style,
  };

  const headerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  };

  const percentageStyle: React.CSSProperties = {
    color,
    fontWeight: 600,
    fontSize: '14px',
  };

  const iconStyle: React.CSSProperties = {
    color,
    fontSize: '14px',
  };

  const messageStyle: React.CSSProperties = {
    color: '#666',
    fontSize: '12px',
    marginTop: '2px',
  };

  const componentStyle: React.CSSProperties = {
    color: '#888',
    fontSize: '11px',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  };

  const borderlineStyle: React.CSSProperties = {
    color: '#888',
    fontSize: '10px',
    fontStyle: 'italic',
    marginTop: '2px',
  };

  return (
    <div style={containerStyle} data-testid="confidence-indicator">
      {component && (
        <span style={componentStyle} data-testid="confidence-component">
          {component}
        </span>
      )}
      <div style={headerStyle}>
        <span style={iconStyle} data-testid="confidence-icon">
          {icon}
        </span>
        {showPercentage && (
          <span style={percentageStyle} data-testid="confidence-percentage">
            {percentage}% confidence
          </span>
        )}
      </div>
      {showMessage && (
        <span style={messageStyle} data-testid="confidence-message">
          {message}
        </span>
      )}
      {isBorderline && (
        <span style={borderlineStyle} data-testid="confidence-borderline">
          (borderline - near threshold)
        </span>
      )}
    </div>
  );
};

// Export thresholds for testing and external use
export { CONFIDENCE_THRESHOLDS, CONFIDENCE_COLORS, CONFIDENCE_MESSAGES };

export default ConfidenceIndicator;

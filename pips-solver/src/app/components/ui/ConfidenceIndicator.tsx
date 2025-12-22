import React from 'react';

// Confidence threshold configuration aligned with cv-service confidence_config.py
// These thresholds determine visual indicators for detection confidence
const CONFIDENCE_THRESHOLDS = {
  high: 0.85,   // User can trust without review
  medium: 0.70, // Suggest review
  // Below 0.70 is considered low confidence
};

// Themed color palette for confidence levels (jade/brass/coral)
const CONFIDENCE_COLORS = {
  high: '#00A878',    // Jade - trustworthy (green)
  medium: '#C9A227',  // Brass - review suggested (gold/amber)
  low: '#FF6B6B',     // Coral - verification required (red)
};

// CSS class names for themed styling
const CONFIDENCE_CLASS_NAMES = {
  high: 'confidence-jade',
  medium: 'confidence-brass',
  low: 'confidence-coral',
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
  /** Size variant for the indicator */
  size?: 'sm' | 'md' | 'lg';
  /** Optional custom className */
  className?: string;
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
 * Gets the themed color for a given confidence level or score.
 * Colors: jade (high), brass (medium), coral (low)
 */
export function getConfidenceColor(confidenceOrLevel: number | ConfidenceLevel): string {
  const level = typeof confidenceOrLevel === 'number'
    ? getConfidenceLevel(confidenceOrLevel)
    : confidenceOrLevel;
  return CONFIDENCE_COLORS[level];
}

/**
 * Gets the CSS class name for a given confidence level or score.
 */
export function getConfidenceClassName(confidenceOrLevel: number | ConfidenceLevel): string {
  const level = typeof confidenceOrLevel === 'number'
    ? getConfidenceLevel(confidenceOrLevel)
    : confidenceOrLevel;
  return CONFIDENCE_CLASS_NAMES[level];
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
 * Get size-specific styles
 */
function getSizeStyles(size: 'sm' | 'md' | 'lg'): {
  fontSize: string;
  iconSize: string;
  gap: string;
  padding: string;
} {
  switch (size) {
    case 'sm':
      return { fontSize: '12px', iconSize: '12px', gap: '4px', padding: '4px 8px' };
    case 'lg':
      return { fontSize: '16px', iconSize: '18px', gap: '10px', padding: '12px 16px' };
    case 'md':
    default:
      return { fontSize: '14px', iconSize: '14px', gap: '8px', padding: '8px 12px' };
  }
}

/**
 * ThemedConfidenceIndicator - Displays visual confidence indicators with themed colors.
 *
 * Color scheme (jade/brass/coral):
 * - high: >= 0.85 (jade #00A878) - User can trust without review
 * - medium: >= 0.70 (brass #C9A227) - Suggest review
 * - low: < 0.70 (coral #FF6B6B) - Requires manual verification
 */
export const ConfidenceIndicator: React.FC<ConfidenceIndicatorProps> = ({
  confidence,
  component,
  showPercentage = true,
  showMessage = true,
  size = 'md',
  className,
  style,
}) => {
  // Clamp confidence to valid range
  const normalizedConfidence = Math.max(0, Math.min(1, confidence));
  const level = getConfidenceLevel(normalizedConfidence);
  const color = getConfidenceColor(level);
  const themeClass = getConfidenceClassName(level);
  const message = getConfidenceMessage(level);
  const icon = getConfidenceIcon(level);
  const isBorderline = isBorderlineConfidence(normalizedConfidence);
  const percentage = Math.round(normalizedConfidence * 100);
  const sizeStyles = getSizeStyles(size);

  const containerStyle: React.CSSProperties = {
    display: 'inline-flex',
    flexDirection: 'column',
    gap: sizeStyles.gap,
    padding: sizeStyles.padding,
    borderRadius: '8px',
    backgroundColor: `${color}15`, // 15% opacity background
    border: `1px solid ${color}30`, // 30% opacity border
    ...style,
  };

  const headerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: sizeStyles.gap,
  };

  const percentageStyle: React.CSSProperties = {
    color,
    fontWeight: 600,
    fontSize: sizeStyles.fontSize,
  };

  const iconStyle: React.CSSProperties = {
    color,
    fontSize: sizeStyles.iconSize,
  };

  const messageStyle: React.CSSProperties = {
    color: '#555',
    fontSize: size === 'sm' ? '10px' : size === 'lg' ? '14px' : '12px',
    marginTop: '2px',
  };

  const componentStyle: React.CSSProperties = {
    color: '#666',
    fontSize: size === 'sm' ? '9px' : size === 'lg' ? '12px' : '11px',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
    fontWeight: 500,
  };

  const borderlineStyle: React.CSSProperties = {
    color: '#888',
    fontSize: size === 'sm' ? '9px' : '10px',
    fontStyle: 'italic',
    marginTop: '2px',
  };

  const combinedClassName = [themeClass, className].filter(Boolean).join(' ');

  return (
    <div
      style={containerStyle}
      className={combinedClassName || undefined}
      data-testid="confidence-indicator"
      data-confidence-level={level}
    >
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

// Export thresholds and colors for testing and external use
export { CONFIDENCE_THRESHOLDS, CONFIDENCE_COLORS, CONFIDENCE_CLASS_NAMES, CONFIDENCE_MESSAGES };

export default ConfidenceIndicator;

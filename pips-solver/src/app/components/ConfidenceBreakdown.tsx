import React from 'react';
import {
  getConfidenceLevel,
  getConfidenceColor,
  isBorderlineConfidence,
  ConfidenceLevel,
  CONFIDENCE_THRESHOLDS,
} from './ConfidenceIndicator';

/**
 * Type definition for confidence breakdown factors from the cv-service API.
 * Each factor is a score from 0.0 to 1.0 indicating quality in that dimension.
 */
export interface ConfidenceFactors {
  saturation?: number;
  area_ratio?: number;
  aspect_ratio?: number;
  relative_size?: number;
  edge_clarity?: number;
  contrast?: number;
  [key: string]: number | undefined;
}

/**
 * Type definition for per-component confidence scores.
 * Maps component names to their confidence scores.
 */
export interface ComponentConfidenceMap {
  geometry_extraction?: number;
  ocr_detection?: number;
  puzzle_detection?: number;
  domino_detection?: number;
  [key: string]: number | undefined;
}

export interface ConfidenceBreakdownProps {
  /** Overall confidence score from 0.0 to 1.0 */
  overallConfidence: number;
  /** Per-component confidence scores */
  componentScores?: ComponentConfidenceMap;
  /** Factor breakdown scores from the API */
  factors?: ConfidenceFactors;
  /** Overall threshold level (high/medium/low) */
  threshold?: ConfidenceLevel;
  /** Whether the overall score is borderline */
  isBorderline?: boolean;
  /** Whether to show factor details */
  showFactors?: boolean;
  /** Whether to show component details */
  showComponents?: boolean;
  /** Optional custom style */
  style?: React.CSSProperties;
}

/**
 * Human-readable labels for detection components.
 */
const COMPONENT_LABELS: Record<string, string> = {
  geometry_extraction: 'Geometry',
  ocr_detection: 'OCR',
  puzzle_detection: 'Puzzle',
  domino_detection: 'Domino',
};

/**
 * Human-readable labels for confidence factors.
 */
const FACTOR_LABELS: Record<string, string> = {
  saturation: 'Saturation',
  area_ratio: 'Area Ratio',
  aspect_ratio: 'Aspect Ratio',
  relative_size: 'Relative Size',
  edge_clarity: 'Edge Clarity',
  contrast: 'Contrast',
};

/**
 * Gets a human-readable label for a component or factor key.
 */
export function getDisplayLabel(key: string): string {
  // Check for component labels first
  if (key in COMPONENT_LABELS) {
    return COMPONENT_LABELS[key];
  }
  // Check for factor labels
  if (key in FACTOR_LABELS) {
    return FACTOR_LABELS[key];
  }
  // Fallback: convert snake_case to Title Case
  return key
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Finds the component with the lowest confidence score.
 * Per spec: "When multiple detection components disagree, use the lowest
 * confidence score and clearly indicate which component is uncertain."
 */
export function findLowestComponent(
  componentScores: ComponentConfidenceMap
): { component: string; score: number } | null {
  let lowest: { component: string; score: number } | null = null;

  for (const [key, score] of Object.entries(componentScores)) {
    if (score !== undefined && (lowest === null || score < lowest.score)) {
      lowest = { component: key, score };
    }
  }

  return lowest;
}

/**
 * Renders a single confidence score bar with color coding.
 */
const ConfidenceBar: React.FC<{
  label: string;
  score: number;
  isLowest?: boolean;
}> = ({ label, score, isLowest = false }) => {
  const normalizedScore = Math.max(0, Math.min(1, score));
  const level = getConfidenceLevel(normalizedScore);
  const color = getConfidenceColor(level);
  const percentage = Math.round(normalizedScore * 100);

  const containerStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    marginBottom: '8px',
  };

  const headerStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    fontSize: '12px',
  };

  const labelStyle: React.CSSProperties = {
    color: isLowest ? color : '#444',
    fontWeight: isLowest ? 600 : 400,
  };

  const percentageStyle: React.CSSProperties = {
    color,
    fontWeight: 500,
  };

  const barContainerStyle: React.CSSProperties = {
    width: '100%',
    height: '6px',
    backgroundColor: '#e5e7eb',
    borderRadius: '3px',
    overflow: 'hidden',
  };

  const barFillStyle: React.CSSProperties = {
    width: `${percentage}%`,
    height: '100%',
    backgroundColor: color,
    borderRadius: '3px',
    transition: 'width 0.3s ease',
  };

  return (
    <div style={containerStyle} data-testid={`confidence-bar-${label.toLowerCase().replace(/\s+/g, '-')}`}>
      <div style={headerStyle}>
        <span style={labelStyle}>
          {label}
          {isLowest && ' (lowest)'}
        </span>
        <span style={percentageStyle}>{percentage}%</span>
      </div>
      <div style={barContainerStyle}>
        <div style={barFillStyle} />
      </div>
    </div>
  );
};

/**
 * ConfidenceBreakdown - Displays detailed per-component confidence scores.
 *
 * Shows breakdown of confidence scores from different detection components
 * (geometry, OCR, puzzle, domino) and individual quality factors.
 *
 * Features:
 * - Color-coded confidence bars for each component
 * - Highlights the lowest component (most uncertain)
 * - Expandable factor details section
 * - Borderline warning for scores near thresholds
 *
 * Per spec: "When multiple detection components disagree, use the lowest
 * confidence score and clearly indicate which component is uncertain."
 */
export const ConfidenceBreakdown: React.FC<ConfidenceBreakdownProps> = ({
  overallConfidence,
  componentScores,
  factors,
  threshold,
  isBorderline: isBorderlineProp,
  showFactors = true,
  showComponents = true,
  style,
}) => {
  // Normalize overall confidence
  const normalizedOverall = Math.max(0, Math.min(1, overallConfidence));
  const overallLevel = threshold || getConfidenceLevel(normalizedOverall);
  const overallColor = getConfidenceColor(overallLevel);
  const overallPercentage = Math.round(normalizedOverall * 100);
  const isBorderline = isBorderlineProp ?? isBorderlineConfidence(normalizedOverall);

  // Find lowest component for highlighting
  const lowestComponent = componentScores ? findLowestComponent(componentScores) : null;

  // Check if there are any component scores to display
  const hasComponentScores = componentScores &&
    Object.values(componentScores).some(score => score !== undefined);

  // Check if there are any factors to display
  const hasFactors = factors &&
    Object.values(factors).some(score => score !== undefined);

  const containerStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '16px',
    borderRadius: '8px',
    backgroundColor: '#fafafa',
    border: '1px solid #e5e7eb',
    ...style,
  };

  const overallStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    paddingBottom: '12px',
    borderBottom: hasComponentScores || hasFactors ? '1px solid #e5e7eb' : 'none',
  };

  const overallHeaderStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  };

  const overallLabelStyle: React.CSSProperties = {
    fontSize: '14px',
    fontWeight: 600,
    color: '#333',
  };

  const overallValueStyle: React.CSSProperties = {
    fontSize: '18px',
    fontWeight: 700,
    color: overallColor,
  };

  const sectionTitleStyle: React.CSSProperties = {
    fontSize: '12px',
    fontWeight: 500,
    color: '#666',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
    marginBottom: '4px',
  };

  const borderlineWarningStyle: React.CSSProperties = {
    fontSize: '11px',
    color: '#888',
    fontStyle: 'italic',
    marginTop: '4px',
  };

  const lowestWarningStyle: React.CSSProperties = {
    fontSize: '11px',
    color: lowestComponent ? getConfidenceColor(lowestComponent.score) : '#888',
    marginTop: '8px',
    padding: '8px',
    backgroundColor: lowestComponent ? `${getConfidenceColor(lowestComponent.score)}15` : 'transparent',
    borderRadius: '4px',
    border: lowestComponent ? `1px solid ${getConfidenceColor(lowestComponent.score)}30` : 'none',
  };

  return (
    <div style={containerStyle} data-testid="confidence-breakdown">
      {/* Overall Confidence */}
      <div style={overallStyle} data-testid="confidence-overall">
        <div style={overallHeaderStyle}>
          <span style={overallLabelStyle}>Overall Confidence</span>
          <span style={overallValueStyle}>{overallPercentage}%</span>
        </div>
        <ConfidenceBar label="Overall" score={normalizedOverall} />
        {isBorderline && (
          <span style={borderlineWarningStyle} data-testid="confidence-borderline-warning">
            Score is near a threshold boundary - results may vary
          </span>
        )}
      </div>

      {/* Component Scores Section */}
      {showComponents && hasComponentScores && (
        <div data-testid="confidence-components">
          <div style={sectionTitleStyle}>Component Scores</div>
          {Object.entries(componentScores!).map(([key, score]) => {
            if (score === undefined) return null;
            const isLowest = lowestComponent?.component === key;
            return (
              <ConfidenceBar
                key={key}
                label={getDisplayLabel(key)}
                score={score}
                isLowest={isLowest}
              />
            );
          })}
          {lowestComponent && lowestComponent.score < CONFIDENCE_THRESHOLDS.medium && (
            <div style={lowestWarningStyle} data-testid="lowest-component-warning">
              <strong>{getDisplayLabel(lowestComponent.component)}</strong> has the lowest confidence.
              Manual verification recommended.
            </div>
          )}
        </div>
      )}

      {/* Factor Details Section */}
      {showFactors && hasFactors && (
        <div data-testid="confidence-factors">
          <div style={sectionTitleStyle}>Quality Factors</div>
          {Object.entries(factors!).map(([key, score]) => {
            if (score === undefined) return null;
            return (
              <ConfidenceBar
                key={key}
                label={getDisplayLabel(key)}
                score={score}
              />
            );
          })}
        </div>
      )}
    </div>
  );
};

export default ConfidenceBreakdown;

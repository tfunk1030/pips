import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ViewStyle } from 'react-native';
import Svg, { Path, Circle } from 'react-native-svg';
import { ConfidenceHint, HintSeverity } from '../../../extraction/types';

interface Props {
  /** Array of confidence hints to display */
  hints: ConfidenceHint[];
  /** Additional container styles */
  style?: ViewStyle;
}

/**
 * List component that displays individual confidence warning hints.
 *
 * Features:
 * - Severity-based icons (warning triangle for high/medium, info circle for low)
 * - Collapsible details showing actionable suggestions
 * - Accessible labels for screen readers
 */
export default function ConfidenceHintsList({ hints, style }: Props) {
  if (hints.length === 0) {
    return null;
  }

  return (
    <View style={[styles.container, style]} accessibilityRole="list">
      {hints.map((hint, index) => (
        <HintItem key={`${hint.stage}-${index}`} hint={hint} />
      ))}
    </View>
  );
}

interface HintItemProps {
  hint: ConfidenceHint;
}

function HintItem({ hint }: HintItemProps) {
  const [expanded, setExpanded] = useState(false);
  const color = getSeverityColor(hint.severity);
  const Icon = hint.severity === 'low' ? InfoIcon : WarningIcon;

  const accessibilityLabel = `${hint.severity} severity warning: ${hint.message}. ${expanded ? hint.suggestion : 'Tap to see suggestion.'}`;

  return (
    <TouchableOpacity
      style={styles.hintItem}
      onPress={() => setExpanded(!expanded)}
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel}
      accessibilityState={{ expanded }}
      accessibilityHint="Double tap to toggle suggestion details"
    >
      <View style={styles.hintHeader}>
        <View style={styles.iconContainer}>
          <Icon color={color} size={18} />
        </View>
        <View style={styles.hintContent}>
          <Text style={[styles.hintMessage, { color }]}>{hint.message}</Text>
          <Text style={styles.hintStage}>{formatStage(hint.stage)}</Text>
        </View>
        <View style={styles.chevronContainer}>
          <ChevronIcon expanded={expanded} color="#9aa5ce" />
        </View>
      </View>
      {expanded && (
        <View style={styles.suggestionContainer}>
          <Text style={styles.suggestionLabel}>Suggestion:</Text>
          <Text style={styles.suggestionText}>{hint.suggestion}</Text>
        </View>
      )}
    </TouchableOpacity>
  );
}

/**
 * Warning triangle icon for high/medium severity hints.
 */
function WarningIcon({ color, size }: { color: string; size: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Path
        d="M12 2L1 21h22L12 2z"
        fill={color}
        fillOpacity={0.15}
        stroke={color}
        strokeWidth={2}
        strokeLinejoin="round"
      />
      <Path d="M12 9v4" stroke={color} strokeWidth={2} strokeLinecap="round" />
      <Circle cx={12} cy={17} r={1} fill={color} />
    </Svg>
  );
}

/**
 * Info circle icon for low severity hints.
 */
function InfoIcon({ color, size }: { color: string; size: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Circle
        cx={12}
        cy={12}
        r={10}
        fill={color}
        fillOpacity={0.15}
        stroke={color}
        strokeWidth={2}
      />
      <Path d="M12 16v-4" stroke={color} strokeWidth={2} strokeLinecap="round" />
      <Circle cx={12} cy={8} r={1} fill={color} />
    </Svg>
  );
}

/**
 * Chevron icon for expand/collapse indication.
 */
function ChevronIcon({ expanded, color }: { expanded: boolean; color: string }) {
  return (
    <Svg width={16} height={16} viewBox="0 0 24 24" fill="none">
      <Path
        d={expanded ? 'M18 15l-6-6-6 6' : 'M6 9l6 6 6-6'}
        stroke={color}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

/**
 * Get color based on hint severity level.
 */
function getSeverityColor(severity: HintSeverity): string {
  switch (severity) {
    case 'high':
      return '#dc2626'; // Red
    case 'medium':
      return '#eab308'; // Yellow
    case 'low':
      return '#22c55e'; // Green
    default:
      return '#9aa5ce'; // Default gray
  }
}

/**
 * Format extraction stage for display.
 */
function formatStage(stage: string): string {
  return stage
    .toLowerCase()
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

const styles = StyleSheet.create({
  container: {
    gap: 8,
  },
  hintItem: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 8,
    overflow: 'hidden',
  },
  hintHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
  },
  iconContainer: {
    marginRight: 10,
  },
  hintContent: {
    flex: 1,
  },
  hintMessage: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 2,
  },
  hintStage: {
    fontSize: 12,
    color: '#9aa5ce',
  },
  chevronContainer: {
    marginLeft: 8,
  },
  suggestionContainer: {
    paddingHorizontal: 12,
    paddingBottom: 12,
    paddingTop: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.02)',
  },
  suggestionLabel: {
    fontSize: 11,
    color: '#9aa5ce',
    marginBottom: 4,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  suggestionText: {
    fontSize: 13,
    color: '#e6e6e6',
    lineHeight: 18,
  },
});

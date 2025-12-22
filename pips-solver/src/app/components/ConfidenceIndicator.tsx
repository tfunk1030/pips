/**
 * Confidence Indicator Component
 * Displays AI extraction confidence scores with visual feedback
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { colors } from '../../theme/tokens';

interface Props {
  label: string;
  confidence?: number;
  compact?: boolean;
}

export default function ConfidenceIndicator({ label, confidence, compact = false }: Props) {
  if (confidence === undefined) {
    return null;
  }

  const getConfidenceColor = (score: number): string => {
    if (score >= 0.9) return colors.semantic.jade; // High confidence
    if (score >= 0.8) return colors.semantic.amber; // Medium confidence
    return colors.semantic.coral; // Low confidence
  };

  const getConfidenceLabel = (score: number): string => {
    if (score >= 0.9) return 'High';
    if (score >= 0.8) return 'Medium';
    return 'Low';
  };

  const color = getConfidenceColor(confidence);
  const confidenceLabel = getConfidenceLabel(confidence);
  const percentage = Math.round(confidence * 100);

  if (compact) {
    return (
      <View style={styles.compactContainer}>
        <View style={[styles.dot, { backgroundColor: color }]} />
        <Text style={styles.compactLabel}>{label}</Text>
        <Text style={styles.compactValue}>{percentage}%</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.label}>{label}</Text>
        <Text style={[styles.confidenceLabel, { color }]}>{confidenceLabel}</Text>
      </View>
      <View style={styles.barContainer}>
        <View style={[styles.barFill, { width: `${percentage}%`, backgroundColor: color }]} />
      </View>
      <Text style={styles.percentage}>{percentage}%</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: 8,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  label: {
    color: colors.text.primary,
    fontSize: 14,
    fontWeight: '500',
  },
  confidenceLabel: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
  },
  barContainer: {
    height: 8,
    backgroundColor: colors.surface.graphite,
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 4,
  },
  barFill: {
    height: '100%',
    borderRadius: 4,
  },
  percentage: {
    color: colors.text.tertiary,
    fontSize: 11,
    textAlign: 'right',
  },
  // Compact styles
  compactContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  compactLabel: {
    color: colors.text.secondary,
    fontSize: 12,
  },
  compactValue: {
    color: colors.text.primary,
    fontSize: 12,
    fontWeight: '600',
  },
});
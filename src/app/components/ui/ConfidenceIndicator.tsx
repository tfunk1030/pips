import React, { useEffect, useRef } from 'react';
import { View, Text, Animated, StyleSheet, ViewStyle } from 'react-native';

interface Props {
  /** Confidence value from 0 to 1 */
  confidence: number;
  /** Compact mode for inline display */
  compact?: boolean;
  /** Additional container styles */
  style?: ViewStyle;
}

/**
 * Visual indicator showing confidence as a percentage with color-coded severity.
 *
 * Color coding:
 * - Green (>80%): High confidence
 * - Yellow (60-80%): Medium confidence
 * - Red (<60%): Low confidence
 */
export default function ConfidenceIndicator({ confidence, compact = false, style }: Props) {
  const animatedValue = useRef(new Animated.Value(confidence)).current;

  useEffect(() => {
    Animated.timing(animatedValue, {
      toValue: confidence,
      duration: 300,
      useNativeDriver: false,
    }).start();
  }, [confidence, animatedValue]);

  const percentage = Math.round(confidence * 100);
  const color = getConfidenceColor(confidence);
  const label = getConfidenceLabel(confidence);

  // Animated background color interpolation
  const backgroundColor = animatedValue.interpolate({
    inputRange: [0, 0.6, 0.8, 1],
    outputRange: ['rgba(220, 38, 38, 0.15)', 'rgba(220, 38, 38, 0.15)', 'rgba(234, 179, 8, 0.15)', 'rgba(34, 197, 94, 0.15)'],
  });

  // Animated width for the progress bar fill
  const fillWidth = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  if (compact) {
    return (
      <View style={[styles.compactContainer, style]}>
        <View style={[styles.compactDot, { backgroundColor: color }]} />
        <Text style={[styles.compactText, { color }]}>{percentage}%</Text>
      </View>
    );
  }

  return (
    <Animated.View style={[styles.container, { backgroundColor }, style]}>
      <View style={styles.header}>
        <Text style={styles.label}>Confidence</Text>
        <Text style={[styles.percentage, { color }]}>{percentage}%</Text>
      </View>
      <View style={styles.progressBar}>
        <Animated.View
          style={[
            styles.progressFill,
            {
              width: fillWidth,
              backgroundColor: color,
            },
          ]}
        />
      </View>
      <Text style={[styles.statusLabel, { color }]}>{label}</Text>
    </Animated.View>
  );
}

/**
 * Get color based on confidence level.
 * Green (>80%), Yellow (60-80%), Red (<60%)
 */
function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) {
    return '#22c55e'; // Green
  } else if (confidence >= 0.6) {
    return '#eab308'; // Yellow
  } else {
    return '#dc2626'; // Red
  }
}

/**
 * Get human-readable label for confidence level.
 */
function getConfidenceLabel(confidence: number): string {
  if (confidence >= 0.8) {
    return 'High Confidence';
  } else if (confidence >= 0.6) {
    return 'Medium Confidence';
  } else {
    return 'Low Confidence';
  }
}

const styles = StyleSheet.create({
  container: {
    padding: 12,
    borderRadius: 8,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  label: {
    color: '#9aa5ce',
    fontSize: 14,
  },
  percentage: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  progressBar: {
    height: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
  statusLabel: {
    fontSize: 12,
    fontWeight: '600',
  },
  compactContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  compactDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 4,
  },
  compactText: {
    fontSize: 14,
    fontWeight: '600',
  },
});

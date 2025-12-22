/**
 * ConfidenceIndicator Component
 *
 * Themed confidence gauge with brass aesthetic
 * Shows AI extraction confidence levels with visual feedback
 */

import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import Animated, {
  useAnimatedStyle,
  withSpring,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import { colors, radii, spacing } from '../../../theme';
import { fontFamilies } from '../../../theme/fonts';
import { springConfigs } from '../../../theme/animations';
import { Text, Label } from './Text';

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

interface ConfidenceIndicatorProps {
  label: string;
  value: number; // 0-1
  showPercentage?: boolean;
  size?: 'compact' | 'full';
  style?: ViewStyle;
}

// ════════════════════════════════════════════════════════════════════════════
// Component
// ════════════════════════════════════════════════════════════════════════════

export function ConfidenceIndicator({
  label,
  value,
  showPercentage = true,
  size = 'full',
  style,
}: ConfidenceIndicatorProps) {
  const percentage = Math.round(value * 100);
  const indicatorColor = getConfidenceColor(value);

  const animatedWidth = useSharedValue(0);

  React.useEffect(() => {
    animatedWidth.value = withSpring(value, springConfigs.gentle);
  }, [value, animatedWidth]);

  const animatedBarStyle = useAnimatedStyle(() => ({
    width: `${animatedWidth.value * 100}%`,
  }));

  const animatedDotStyle = useAnimatedStyle(() => ({
    left: `${animatedWidth.value * 100}%`,
  }));

  if (size === 'compact') {
    return (
      <View style={[styles.compactContainer, style]}>
        <View style={[styles.compactDot, { backgroundColor: indicatorColor }]} />
        <Label size="small" style={styles.compactLabel}>
          {label}
        </Label>
        {showPercentage && (
          <Text variant="monoSmall" style={{ color: indicatorColor }}>
            {percentage}%
          </Text>
        )}
      </View>
    );
  }

  return (
    <View style={[styles.container, style]}>
      <View style={styles.header}>
        <Label size="medium">{label}</Label>
        {showPercentage && (
          <Text variant="mono" style={{ color: indicatorColor }}>
            {percentage}%
          </Text>
        )}
      </View>

      <View style={styles.track}>
        {/* Background track */}
        <View style={styles.trackBackground} />

        {/* Filled bar */}
        <Animated.View
          style={[
            styles.trackFill,
            { backgroundColor: indicatorColor },
            animatedBarStyle,
          ]}
        />

        {/* Brass dot indicator */}
        <Animated.View style={[styles.dotContainer, animatedDotStyle]}>
          <View
            style={[
              styles.dot,
              {
                backgroundColor: indicatorColor,
                shadowColor: indicatorColor,
              },
            ]}
          />
        </Animated.View>
      </View>
    </View>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ════════════════════════════════════════════════════════════════════════════

function getConfidenceColor(value: number): string {
  if (value >= 0.9) return colors.semantic.jade;
  if (value >= 0.8) return colors.accent.brass;
  return colors.semantic.coral;
}

// ════════════════════════════════════════════════════════════════════════════
// Styles
// ════════════════════════════════════════════════════════════════════════════

const styles = StyleSheet.create({
  container: {
    marginVertical: spacing[2],
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing[2],
  },
  track: {
    height: 6,
    position: 'relative',
  },
  trackBackground: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    backgroundColor: colors.surface.graphite,
    borderRadius: radii.full,
  },
  trackFill: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    borderRadius: radii.full,
    minWidth: 6,
  },
  dotContainer: {
    position: 'absolute',
    top: -5,
    marginLeft: -8,
  },
  dot: {
    width: 16,
    height: 16,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: colors.surface.charcoal,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 4,
    elevation: 4,
  },
  compactContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing[2],
  },
  compactDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  compactLabel: {
    flex: 1,
  },
});

// ════════════════════════════════════════════════════════════════════════════
// Confidence Summary Component
// ════════════════════════════════════════════════════════════════════════════

interface ConfidenceSummaryProps {
  grid?: number;
  regions?: number;
  constraints?: number;
  dominoes?: number;
  style?: ViewStyle;
}

export function ConfidenceSummary({
  grid,
  regions,
  constraints,
  dominoes,
  style,
}: ConfidenceSummaryProps) {
  return (
    <View style={[summaryStyles.container, style]}>
      {grid !== undefined && (
        <ConfidenceIndicator label="Grid" value={grid} />
      )}
      {regions !== undefined && (
        <ConfidenceIndicator label="Regions" value={regions} />
      )}
      {constraints !== undefined && (
        <ConfidenceIndicator label="Constraints" value={constraints} />
      )}
      {dominoes !== undefined && (
        <ConfidenceIndicator label="Dominoes" value={dominoes} />
      )}
    </View>
  );
}

const summaryStyles = StyleSheet.create({
  container: {
    gap: spacing[1],
  },
});

export default ConfidenceIndicator;

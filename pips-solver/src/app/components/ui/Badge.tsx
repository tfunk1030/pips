/**
 * Badge Component
 *
 * Small status indicators and labels
 * Uses semantic colors for status and brass for accents
 */

import React from 'react';
import { StyleSheet, Text, View, ViewStyle, TextStyle } from 'react-native';
import { colors, radii, spacing } from '../../../theme';
import { fontFamilies } from '../../../theme/fonts';

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

type BadgeVariant = 'default' | 'success' | 'warning' | 'error' | 'info' | 'accent';
type BadgeSize = 'small' | 'medium';

interface BadgeProps {
  label: string;
  variant?: BadgeVariant;
  size?: BadgeSize;
  icon?: React.ReactNode;
  style?: ViewStyle;
}

// ════════════════════════════════════════════════════════════════════════════
// Component
// ════════════════════════════════════════════════════════════════════════════

export function Badge({
  label,
  variant = 'default',
  size = 'medium',
  icon,
  style,
}: BadgeProps) {
  const variantStyles = getVariantStyles(variant);
  const sizeStyles = getSizeStyles(size);

  return (
    <View style={[styles.base, variantStyles.container, sizeStyles.container, style]}>
      {icon && <View style={styles.icon}>{icon}</View>}
      <Text style={[styles.text, variantStyles.text, sizeStyles.text]}>{label}</Text>
    </View>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Variant Styles
// ════════════════════════════════════════════════════════════════════════════

function getVariantStyles(variant: BadgeVariant): {
  container: ViewStyle;
  text: TextStyle;
} {
  switch (variant) {
    case 'default':
      return {
        container: {
          backgroundColor: colors.surface.graphite,
        },
        text: {
          color: colors.text.secondary,
        },
      };
    case 'success':
      return {
        container: {
          backgroundColor: `${colors.semantic.jade}20`,
        },
        text: {
          color: colors.semantic.jade,
        },
      };
    case 'warning':
      return {
        container: {
          backgroundColor: `${colors.semantic.amber}20`,
        },
        text: {
          color: colors.semantic.amber,
        },
      };
    case 'error':
      return {
        container: {
          backgroundColor: `${colors.semantic.coral}20`,
        },
        text: {
          color: colors.semantic.coral,
        },
      };
    case 'info':
      return {
        container: {
          backgroundColor: `${colors.accent.copper}20`,
        },
        text: {
          color: colors.accent.copper,
        },
      };
    case 'accent':
      return {
        container: {
          backgroundColor: colors.accent.brass,
        },
        text: {
          color: colors.surface.obsidian,
        },
      };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Size Styles
// ════════════════════════════════════════════════════════════════════════════

function getSizeStyles(size: BadgeSize): {
  container: ViewStyle;
  text: TextStyle;
} {
  switch (size) {
    case 'small':
      return {
        container: {
          paddingVertical: spacing[1],
          paddingHorizontal: spacing[2],
          borderRadius: radii.sm,
        },
        text: {
          fontSize: 10,
          letterSpacing: 0.5,
        },
      };
    case 'medium':
      return {
        container: {
          paddingVertical: spacing[1] + 2,
          paddingHorizontal: spacing[3],
          borderRadius: radii.md,
        },
        text: {
          fontSize: 12,
          letterSpacing: 0.5,
        },
      };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Styles
// ════════════════════════════════════════════════════════════════════════════

const styles = StyleSheet.create({
  base: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
  },
  text: {
    fontFamily: fontFamilies.bodyMedium,
    textTransform: 'uppercase',
  },
  icon: {
    marginRight: spacing[1],
  },
});

// ════════════════════════════════════════════════════════════════════════════
// Status Dot Component
// ════════════════════════════════════════════════════════════════════════════

interface StatusDotProps {
  status: 'success' | 'warning' | 'error' | 'neutral';
  size?: number;
  pulse?: boolean;
  style?: ViewStyle;
}

export function StatusDot({
  status,
  size = 8,
  pulse = false,
  style,
}: StatusDotProps) {
  const dotColor = {
    success: colors.semantic.jade,
    warning: colors.semantic.amber,
    error: colors.semantic.coral,
    neutral: colors.surface.ash,
  }[status];

  return (
    <View
      style={[
        {
          width: size,
          height: size,
          borderRadius: size / 2,
          backgroundColor: dotColor,
        },
        pulse && {
          shadowColor: dotColor,
          shadowOffset: { width: 0, height: 0 },
          shadowOpacity: 0.6,
          shadowRadius: 4,
        },
        style,
      ]}
    />
  );
}

export default Badge;

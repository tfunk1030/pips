/**
 * Card Component
 *
 * Themed card container with subtle elevation and optional press interaction
 * Uses slate background with optional brass accent border
 */

import React, { useCallback } from 'react';
import {
  Pressable,
  PressableProps,
  StyleSheet,
  View,
  ViewStyle,
} from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { colors, radii, spacing, shadows } from '../../../theme';
import { springConfigs, timingConfigs } from '../../../theme/animations';

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

type CardVariant = 'default' | 'elevated' | 'outlined' | 'accent';

interface CardProps {
  children: React.ReactNode;
  variant?: CardVariant;
  onPress?: () => void;
  disabled?: boolean;
  style?: ViewStyle;
  contentStyle?: ViewStyle;
}

// ════════════════════════════════════════════════════════════════════════════
// Animated Pressable
// ════════════════════════════════════════════════════════════════════════════

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

// ════════════════════════════════════════════════════════════════════════════
// Component
// ════════════════════════════════════════════════════════════════════════════

export function Card({
  children,
  variant = 'default',
  onPress,
  disabled = false,
  style,
  contentStyle,
}: CardProps) {
  const scale = useSharedValue(1);
  const borderOpacity = useSharedValue(variant === 'accent' ? 1 : 0);

  const handlePressIn = useCallback(() => {
    if (onPress && !disabled) {
      scale.value = withTiming(0.98, timingConfigs.fast);
      if (variant !== 'accent') {
        borderOpacity.value = withTiming(1, timingConfigs.fast);
      }
    }
  }, [scale, borderOpacity, onPress, disabled, variant]);

  const handlePressOut = useCallback(() => {
    scale.value = withSpring(1, springConfigs.snappy);
    if (variant !== 'accent') {
      borderOpacity.value = withTiming(0, timingConfigs.normal);
    }
  }, [scale, borderOpacity, variant]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const animatedBorderStyle = useAnimatedStyle(() => ({
    borderColor:
      variant === 'accent'
        ? colors.accent.brass
        : `rgba(232, 197, 71, ${borderOpacity.value * 0.5})`,
  }));

  const variantStyles = getVariantStyles(variant);

  const Container = onPress ? AnimatedPressable : View;
  const containerProps = onPress
    ? {
        onPress,
        onPressIn: handlePressIn,
        onPressOut: handlePressOut,
        disabled,
      }
    : {};

  return (
    <Container
      {...containerProps}
      style={[
        styles.base,
        variantStyles,
        onPress && animatedStyle,
        onPress && animatedBorderStyle,
        disabled && styles.disabled,
        style,
      ]}
    >
      <View style={[styles.content, contentStyle]}>{children}</View>
    </Container>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Variant Styles
// ════════════════════════════════════════════════════════════════════════════

function getVariantStyles(variant: CardVariant): ViewStyle {
  switch (variant) {
    case 'default':
      return {
        backgroundColor: colors.surface.slate,
        borderWidth: 1,
        borderColor: colors.surface.ash,
      };
    case 'elevated':
      return {
        backgroundColor: colors.surface.slate,
        ...shadows.lg,
      };
    case 'outlined':
      return {
        backgroundColor: 'transparent',
        borderWidth: 1,
        borderColor: colors.surface.ash,
      };
    case 'accent':
      return {
        backgroundColor: colors.surface.slate,
        borderWidth: 2,
        borderColor: colors.accent.brass,
        ...shadows.glow,
      };
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Styles
// ════════════════════════════════════════════════════════════════════════════

const styles = StyleSheet.create({
  base: {
    borderRadius: radii.lg,
    overflow: 'hidden',
  },
  content: {
    padding: spacing[4],
  },
  disabled: {
    opacity: 0.5,
  },
});

// ════════════════════════════════════════════════════════════════════════════
// Card Header Component
// ════════════════════════════════════════════════════════════════════════════

interface CardHeaderProps {
  children: React.ReactNode;
  style?: ViewStyle;
}

export function CardHeader({ children, style }: CardHeaderProps) {
  return <View style={[cardHeaderStyles.header, style]}>{children}</View>;
}

const cardHeaderStyles = StyleSheet.create({
  header: {
    paddingBottom: spacing[3],
    marginBottom: spacing[3],
    borderBottomWidth: 1,
    borderBottomColor: colors.surface.ash,
  },
});

// ════════════════════════════════════════════════════════════════════════════
// Card Footer Component
// ════════════════════════════════════════════════════════════════════════════

interface CardFooterProps {
  children: React.ReactNode;
  style?: ViewStyle;
}

export function CardFooter({ children, style }: CardFooterProps) {
  return <View style={[cardFooterStyles.footer, style]}>{children}</View>;
}

const cardFooterStyles = StyleSheet.create({
  footer: {
    paddingTop: spacing[3],
    marginTop: spacing[3],
    borderTopWidth: 1,
    borderTopColor: colors.surface.ash,
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: spacing[2],
  },
});

export default Card;

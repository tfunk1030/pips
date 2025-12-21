/**
 * Button Component
 *
 * Themed button with press animations and multiple variants
 * Uses brass accent colors with tactile feedback
 */

import React, { useCallback } from 'react';
import {
  Pressable,
  PressableProps,
  StyleSheet,
  Text,
  View,
  ViewStyle,
  TextStyle,
} from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';
import { colors, radii, spacing, shadows } from '../../../theme';
import { fontFamilies, textStyles } from '../../../theme/fonts';
import { springConfigs, timingConfigs } from '../../../theme/animations';

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'success';
type ButtonSize = 'small' | 'medium' | 'large';

interface ButtonProps extends Omit<PressableProps, 'style'> {
  title: string;
  variant?: ButtonVariant;
  size?: ButtonSize;
  fullWidth?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  loading?: boolean;
  style?: ViewStyle;
}

// ════════════════════════════════════════════════════════════════════════════
// Animated Pressable
// ════════════════════════════════════════════════════════════════════════════

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

// ════════════════════════════════════════════════════════════════════════════
// Component
// ════════════════════════════════════════════════════════════════════════════

export function Button({
  title,
  variant = 'primary',
  size = 'medium',
  fullWidth = false,
  leftIcon,
  rightIcon,
  loading = false,
  disabled,
  style,
  onPressIn,
  onPressOut,
  ...pressableProps
}: ButtonProps) {
  const scale = useSharedValue(1);
  const opacity = useSharedValue(1);

  const handlePressIn = useCallback(
    (e: any) => {
      scale.value = withTiming(0.97, timingConfigs.fast);
      opacity.value = withTiming(0.9, timingConfigs.fast);
      onPressIn?.(e);
    },
    [scale, opacity, onPressIn]
  );

  const handlePressOut = useCallback(
    (e: any) => {
      scale.value = withSpring(1, springConfigs.snappy);
      opacity.value = withTiming(1, timingConfigs.fast);
      onPressOut?.(e);
    },
    [scale, opacity, onPressOut]
  );

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
    opacity: opacity.value,
  }));

  const variantStyles = getVariantStyles(variant);
  const sizeStyles = getSizeStyles(size);

  const isDisabled = disabled || loading;

  return (
    <AnimatedPressable
      {...pressableProps}
      disabled={isDisabled}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      style={[
        styles.base,
        variantStyles.container,
        sizeStyles.container,
        fullWidth && styles.fullWidth,
        isDisabled && styles.disabled,
        animatedStyle,
        style,
      ]}
    >
      <View style={styles.content}>
        {leftIcon && <View style={styles.leftIcon}>{leftIcon}</View>}
        <Text
          style={[
            styles.text,
            variantStyles.text,
            sizeStyles.text,
            isDisabled && styles.disabledText,
          ]}
        >
          {loading ? 'Loading...' : title}
        </Text>
        {rightIcon && <View style={styles.rightIcon}>{rightIcon}</View>}
      </View>
    </AnimatedPressable>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Variant Styles
// ════════════════════════════════════════════════════════════════════════════

function getVariantStyles(variant: ButtonVariant): {
  container: ViewStyle;
  text: TextStyle;
} {
  switch (variant) {
    case 'primary':
      return {
        container: {
          backgroundColor: colors.accent.brass,
          ...shadows.md,
        },
        text: {
          color: colors.surface.obsidian,
        },
      };
    case 'secondary':
      return {
        container: {
          backgroundColor: colors.surface.slate,
          borderWidth: 1,
          borderColor: colors.surface.ash,
        },
        text: {
          color: colors.text.primary,
        },
      };
    case 'ghost':
      return {
        container: {
          backgroundColor: 'transparent',
        },
        text: {
          color: colors.accent.brass,
        },
      };
    case 'danger':
      return {
        container: {
          backgroundColor: colors.semantic.coral,
          ...shadows.md,
        },
        text: {
          color: colors.text.primary,
        },
      };
    case 'success':
      return {
        container: {
          backgroundColor: colors.semantic.jade,
          ...shadows.md,
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

function getSizeStyles(size: ButtonSize): {
  container: ViewStyle;
  text: TextStyle;
} {
  switch (size) {
    case 'small':
      return {
        container: {
          paddingVertical: spacing[2],
          paddingHorizontal: spacing[3],
          borderRadius: radii.md,
        },
        text: {
          ...textStyles.buttonSmall,
        },
      };
    case 'medium':
      return {
        container: {
          paddingVertical: spacing[3],
          paddingHorizontal: spacing[4],
          borderRadius: radii.lg,
        },
        text: {
          ...textStyles.buttonMedium,
        },
      };
    case 'large':
      return {
        container: {
          paddingVertical: spacing[4],
          paddingHorizontal: spacing[6],
          borderRadius: radii.lg,
        },
        text: {
          ...textStyles.buttonLarge,
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
    justifyContent: 'center',
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    fontFamily: fontFamilies.bodySemiBold,
    textAlign: 'center',
  },
  fullWidth: {
    width: '100%',
  },
  disabled: {
    opacity: 0.5,
  },
  disabledText: {
    opacity: 0.7,
  },
  leftIcon: {
    marginRight: spacing[2],
  },
  rightIcon: {
    marginLeft: spacing[2],
  },
});

export default Button;

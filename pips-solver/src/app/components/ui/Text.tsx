/**
 * Text Component
 *
 * Themed text with predefined style variants
 * Automatically applies correct font family and colors
 */

import React from 'react';
import { Text as RNText, TextProps as RNTextProps, StyleSheet } from 'react-native';
import { textStyles, TextStyleName } from '../../../theme/fonts';
import { colors } from '../../../theme';

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

interface TextProps extends RNTextProps {
  variant?: TextStyleName;
  color?: keyof typeof colors.text | string;
  align?: 'left' | 'center' | 'right';
  children: React.ReactNode;
}

// ════════════════════════════════════════════════════════════════════════════
// Component
// ════════════════════════════════════════════════════════════════════════════

export function Text({
  variant = 'bodyMedium',
  color,
  align,
  style,
  children,
  ...props
}: TextProps) {
  const variantStyle = textStyles[variant];

  const colorStyle = color
    ? { color: color in colors.text ? colors.text[color as keyof typeof colors.text] : color }
    : {};

  const alignStyle = align ? { textAlign: align } : {};

  return (
    <RNText
      style={[variantStyle, colorStyle, alignStyle, style]}
      {...props}
    >
      {children}
    </RNText>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Convenience Components
// ════════════════════════════════════════════════════════════════════════════

export function DisplayText(props: Omit<TextProps, 'variant'> & { size?: 'large' | 'medium' | 'small' }) {
  const { size = 'medium', ...rest } = props;
  const variant = {
    large: 'displayLarge',
    medium: 'displayMedium',
    small: 'displaySmall',
  }[size] as TextStyleName;

  return <Text variant={variant} {...rest} />;
}

export function Heading(props: Omit<TextProps, 'variant'> & { size?: 'large' | 'medium' | 'small' }) {
  const { size = 'medium', ...rest } = props;
  const variant = {
    large: 'headingLarge',
    medium: 'headingMedium',
    small: 'headingSmall',
  }[size] as TextStyleName;

  return <Text variant={variant} {...rest} />;
}

export function Body(props: Omit<TextProps, 'variant'> & { size?: 'large' | 'medium' | 'small' }) {
  const { size = 'medium', ...rest } = props;
  const variant = {
    large: 'bodyLarge',
    medium: 'bodyMedium',
    small: 'bodySmall',
  }[size] as TextStyleName;

  return <Text variant={variant} {...rest} />;
}

export function Label(props: Omit<TextProps, 'variant'> & { size?: 'large' | 'medium' | 'small' }) {
  const { size = 'medium', ...rest } = props;
  const variant = {
    large: 'labelLarge',
    medium: 'labelMedium',
    small: 'labelSmall',
  }[size] as TextStyleName;

  return <Text variant={variant} {...rest} />;
}

export function Mono(props: Omit<TextProps, 'variant'> & { small?: boolean }) {
  const { small = false, ...rest } = props;
  return <Text variant={small ? 'monoSmall' : 'mono'} {...rest} />;
}

export default Text;

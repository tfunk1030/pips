/**
 * Divider Component
 *
 * Horizontal or vertical separator line
 */

import React from 'react';
import { View, ViewStyle, StyleSheet } from 'react-native';
import { colors, spacing } from '../../../theme';

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

interface DividerProps {
  orientation?: 'horizontal' | 'vertical';
  spacing?: keyof typeof spacing | number;
  color?: string;
  style?: ViewStyle;
}

// ════════════════════════════════════════════════════════════════════════════
// Component
// ════════════════════════════════════════════════════════════════════════════

export function Divider({
  orientation = 'horizontal',
  spacing: spacingProp = 3,
  color = colors.surface.ash,
  style,
}: DividerProps) {
  const spacingValue =
    typeof spacingProp === 'number' && spacingProp in spacing
      ? spacing[spacingProp as keyof typeof spacing]
      : typeof spacingProp === 'number'
      ? spacingProp
      : spacing[spacingProp];

  const dividerStyle: ViewStyle =
    orientation === 'horizontal'
      ? {
          height: 1,
          backgroundColor: color,
          marginVertical: spacingValue,
        }
      : {
          width: 1,
          backgroundColor: color,
          marginHorizontal: spacingValue,
        };

  return <View style={[dividerStyle, style]} />;
}

export default Divider;

/**
 * Font Configuration
 *
 * Uses expo-google-fonts for custom typography:
 * - Playfair Display: Editorial display font for titles
 * - DM Sans: Geometric but warm body text
 * - JetBrains Mono: Code and data display
 */

import {
  useFonts,
  PlayfairDisplay_400Regular,
  PlayfairDisplay_500Medium,
  PlayfairDisplay_600SemiBold,
  PlayfairDisplay_700Bold,
} from '@expo-google-fonts/playfair-display';

import {
  DMSans_400Regular,
  DMSans_500Medium,
  DMSans_600SemiBold,
  DMSans_700Bold,
} from '@expo-google-fonts/dm-sans';

import {
  JetBrainsMono_400Regular,
  JetBrainsMono_500Medium,
} from '@expo-google-fonts/jetbrains-mono';

// ════════════════════════════════════════════════════════════════════════════
// Font Loading Hook
// ════════════════════════════════════════════════════════════════════════════

export function useAppFonts() {
  const [fontsLoaded, fontError] = useFonts({
    // Playfair Display - Display/Title font
    PlayfairDisplay_400Regular,
    PlayfairDisplay_500Medium,
    PlayfairDisplay_600SemiBold,
    PlayfairDisplay_700Bold,

    // DM Sans - Body font
    DMSans_400Regular,
    DMSans_500Medium,
    DMSans_600SemiBold,
    DMSans_700Bold,

    // JetBrains Mono - Monospace font
    JetBrainsMono_400Regular,
    JetBrainsMono_500Medium,
  });

  return { fontsLoaded, fontError };
}

// ════════════════════════════════════════════════════════════════════════════
// Font Family Names
// ════════════════════════════════════════════════════════════════════════════

export const fontFamilies = {
  // Display fonts (Playfair Display)
  displayRegular: 'PlayfairDisplay_400Regular',
  displayMedium: 'PlayfairDisplay_500Medium',
  displaySemiBold: 'PlayfairDisplay_600SemiBold',
  displayBold: 'PlayfairDisplay_700Bold',

  // Body fonts (DM Sans)
  bodyRegular: 'DMSans_400Regular',
  bodyMedium: 'DMSans_500Medium',
  bodySemiBold: 'DMSans_600SemiBold',
  bodyBold: 'DMSans_700Bold',

  // Mono fonts (JetBrains Mono)
  monoRegular: 'JetBrainsMono_400Regular',
  monoMedium: 'JetBrainsMono_500Medium',
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Text Style Presets
// ════════════════════════════════════════════════════════════════════════════

import { TextStyle } from 'react-native';
import { colors, typography } from './tokens';

export const textStyles: Record<string, TextStyle> = {
  // Display styles (titles, hero text)
  displayLarge: {
    fontFamily: fontFamilies.displayBold,
    fontSize: typography.sizes['5xl'],
    lineHeight: typography.sizes['5xl'] * typography.lineHeights.tight,
    letterSpacing: typography.letterSpacing.tight,
    color: colors.text.primary,
  },
  displayMedium: {
    fontFamily: fontFamilies.displayBold,
    fontSize: typography.sizes['4xl'],
    lineHeight: typography.sizes['4xl'] * typography.lineHeights.tight,
    letterSpacing: typography.letterSpacing.tight,
    color: colors.text.primary,
  },
  displaySmall: {
    fontFamily: fontFamilies.displaySemiBold,
    fontSize: typography.sizes['3xl'],
    lineHeight: typography.sizes['3xl'] * typography.lineHeights.tight,
    letterSpacing: typography.letterSpacing.normal,
    color: colors.text.primary,
  },

  // Heading styles
  headingLarge: {
    fontFamily: fontFamilies.bodySemiBold,
    fontSize: typography.sizes['2xl'],
    lineHeight: typography.sizes['2xl'] * typography.lineHeights.normal,
    letterSpacing: typography.letterSpacing.normal,
    color: colors.text.primary,
  },
  headingMedium: {
    fontFamily: fontFamilies.bodySemiBold,
    fontSize: typography.sizes.xl,
    lineHeight: typography.sizes.xl * typography.lineHeights.normal,
    color: colors.text.primary,
  },
  headingSmall: {
    fontFamily: fontFamilies.bodySemiBold,
    fontSize: typography.sizes.lg,
    lineHeight: typography.sizes.lg * typography.lineHeights.normal,
    color: colors.text.primary,
  },

  // Body styles
  bodyLarge: {
    fontFamily: fontFamilies.bodyRegular,
    fontSize: typography.sizes.lg,
    lineHeight: typography.sizes.lg * typography.lineHeights.relaxed,
    color: colors.text.primary,
  },
  bodyMedium: {
    fontFamily: fontFamilies.bodyRegular,
    fontSize: typography.sizes.base,
    lineHeight: typography.sizes.base * typography.lineHeights.relaxed,
    color: colors.text.primary,
  },
  bodySmall: {
    fontFamily: fontFamilies.bodyRegular,
    fontSize: typography.sizes.sm,
    lineHeight: typography.sizes.sm * typography.lineHeights.relaxed,
    color: colors.text.secondary,
  },

  // Label styles
  labelLarge: {
    fontFamily: fontFamilies.bodyMedium,
    fontSize: typography.sizes.base,
    lineHeight: typography.sizes.base * typography.lineHeights.normal,
    letterSpacing: typography.letterSpacing.wide,
    color: colors.text.primary,
  },
  labelMedium: {
    fontFamily: fontFamilies.bodyMedium,
    fontSize: typography.sizes.sm,
    lineHeight: typography.sizes.sm * typography.lineHeights.normal,
    letterSpacing: typography.letterSpacing.wide,
    color: colors.text.secondary,
  },
  labelSmall: {
    fontFamily: fontFamilies.bodyMedium,
    fontSize: typography.sizes.xs,
    lineHeight: typography.sizes.xs * typography.lineHeights.normal,
    letterSpacing: typography.letterSpacing.extraWide,
    textTransform: 'uppercase',
    color: colors.text.tertiary,
  },

  // Mono styles (for code, data)
  mono: {
    fontFamily: fontFamilies.monoRegular,
    fontSize: typography.sizes.sm,
    lineHeight: typography.sizes.sm * typography.lineHeights.relaxed,
    color: colors.text.primary,
  },
  monoSmall: {
    fontFamily: fontFamilies.monoRegular,
    fontSize: typography.sizes.xs,
    lineHeight: typography.sizes.xs * typography.lineHeights.relaxed,
    color: colors.text.secondary,
  },

  // Button styles
  buttonLarge: {
    fontFamily: fontFamilies.bodySemiBold,
    fontSize: typography.sizes.lg,
    lineHeight: typography.sizes.lg * typography.lineHeights.normal,
    letterSpacing: typography.letterSpacing.wide,
  },
  buttonMedium: {
    fontFamily: fontFamilies.bodySemiBold,
    fontSize: typography.sizes.base,
    lineHeight: typography.sizes.base * typography.lineHeights.normal,
    letterSpacing: typography.letterSpacing.wide,
  },
  buttonSmall: {
    fontFamily: fontFamilies.bodyMedium,
    fontSize: typography.sizes.sm,
    lineHeight: typography.sizes.sm * typography.lineHeights.normal,
    letterSpacing: typography.letterSpacing.wide,
  },
};

export type TextStyleName = keyof typeof textStyles;

/**
 * Styled Utilities
 * Helper functions for creating themed styles
 */

import { StyleSheet, ViewStyle, TextStyle, ImageStyle } from 'react-native';
import { theme, Theme } from './tokens';

type NamedStyles<T> = { [P in keyof T]: ViewStyle | TextStyle | ImageStyle };
type StyleFactory<T> = (theme: Theme) => T;

/**
 * Create styles with access to theme tokens
 *
 * @example
 * const styles = createStyles((theme) => ({
 *   container: {
 *     backgroundColor: theme.colors.surface.charcoal,
 *     padding: theme.spacing[4],
 *   },
 * }));
 */
export function createStyles<T extends NamedStyles<T>>(
  factory: StyleFactory<T>
): T {
  return StyleSheet.create(factory(theme));
}

/**
 * Merge multiple style objects
 */
export function mergeStyles<T extends ViewStyle | TextStyle | ImageStyle>(
  ...styles: (T | undefined | null | false)[]
): T {
  return Object.assign({}, ...styles.filter(Boolean)) as T;
}

/**
 * Create a styled component wrapper (simplified version)
 * For more complex needs, use createStyles directly
 */
export function styled<T extends NamedStyles<T>>(
  factory: StyleFactory<T>
): { styles: T; theme: Theme } {
  return {
    styles: StyleSheet.create(factory(theme)),
    theme,
  };
}

/**
 * Get spacing value with fallback
 */
export function getSpacing(key: keyof typeof theme.spacing): number {
  return theme.spacing[key];
}

/**
 * Get color with path notation
 * @example getColor('surface.charcoal') => '#1A1A1F'
 */
export function getColor(path: string): string {
  const parts = path.split('.');
  let value: any = theme.colors;

  for (const part of parts) {
    if (value && typeof value === 'object' && part in value) {
      value = value[part];
    } else {
      console.warn(`Color path not found: ${path}`);
      return '#FF00FF'; // Magenta for debugging
    }
  }

  return value as string;
}

/**
 * Common style patterns
 */
export const commonStyles = createStyles((t) => ({
  // Layout
  flex1: {
    flex: 1,
  },
  flexRow: {
    flexDirection: 'row',
  },
  flexRowCenter: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  flexCenter: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  flexBetween: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },

  // Backgrounds
  bgObsidian: {
    backgroundColor: t.colors.surface.obsidian,
  },
  bgCharcoal: {
    backgroundColor: t.colors.surface.charcoal,
  },
  bgSlate: {
    backgroundColor: t.colors.surface.slate,
  },

  // Text
  textPrimary: {
    color: t.colors.text.primary,
  },
  textSecondary: {
    color: t.colors.text.secondary,
  },

  // Cards
  card: {
    backgroundColor: t.colors.surface.slate,
    borderRadius: t.radii.lg,
    padding: t.spacing[4],
    ...t.shadows.md,
  },

  // Dividers
  divider: {
    height: 1,
    backgroundColor: t.colors.surface.ash,
    marginVertical: t.spacing[3],
  },
}));

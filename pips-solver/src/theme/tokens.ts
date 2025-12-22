/**
 * Design System Tokens - Tactile Game Table Theme
 *
 * A warm brass/copper accent palette inspired by classic game tables.
 * This theme provides a cohesive visual language for the Pips Solver app.
 */

// ═══════════════════════════════════════════════════════════════════════════
// Color Primitives
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Raw color values - not for direct use in components.
 * Use semantic tokens from the `tokens` export instead.
 */
const palette = {
  // Brass & Copper tones
  brass: {
    50: '#FDF8E7',
    100: '#F9EDCA',
    200: '#EFD79A',
    300: '#E5C16A',
    400: '#D4A84A',
    500: '#C89B3C', // Primary brass
    600: '#B8872F',
    700: '#9A6F25',
    800: '#7C581D',
    900: '#5E4215',
  },

  // Teal complement (for AI actions - harmonizes with brass)
  teal: {
    50: '#E6FAF8',
    100: '#B3F0EB',
    200: '#80E6DE',
    300: '#4DDCD1',
    400: '#26D2C6',
    500: '#1ABC9C', // Primary teal
    600: '#169F84',
    700: '#11826C',
    800: '#0D6554',
    900: '#08483C',
  },

  // Neutral grays (warm undertone)
  gray: {
    50: '#FAFAFA',
    100: '#F0F0EE',
    200: '#E0DFDC',
    300: '#C8C7C3',
    400: '#A8A6A0',
    500: '#888682',
    600: '#666460',
    700: '#4A4845',
    800: '#333230',
    900: '#222120',
    950: '#1A1918',
  },

  // Pure values
  white: '#FFFFFF',
  black: '#000000',
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// Semantic Color Tokens
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Semantic color tokens for use in components.
 * These abstract the raw colors into meaningful categories.
 */
const colors = {
  /**
   * Accent colors for primary actions and highlights
   */
  accent: {
    /** Primary brass accent for buttons and interactive elements */
    primary: palette.brass[500],
    /** Lighter variant for hover/highlight states */
    primaryLight: palette.brass[400],
    /** Darker variant for pressed states */
    primaryDark: palette.brass[600],
    /** AI-related actions - teal provides good contrast with brass */
    ai: palette.teal[500],
    /** AI lighter variant */
    aiLight: palette.teal[400],
    /** AI darker variant */
    aiDark: palette.teal[600],
  },

  /**
   * Surface colors for backgrounds and containers
   */
  surface: {
    /** Main background (darkest) */
    background: palette.black,
    /** Primary surface for cards and containers */
    primary: palette.gray[950],
    /** Secondary surface for nested elements */
    secondary: palette.gray[900],
    /** Elevated surface for buttons and controls */
    elevated: palette.gray[800],
    /** Dark surface for overlays */
    dark: palette.black,
  },

  /**
   * Text colors for typography
   */
  text: {
    /** Primary text - high contrast */
    primary: palette.white,
    /** Secondary text - medium contrast */
    secondary: palette.gray[300],
    /** Muted text - low contrast for hints/labels */
    muted: palette.gray[500],
    /** Inverse text for light backgrounds */
    inverse: palette.white,
    /** Link/action text color */
    link: palette.brass[500],
  },

  /**
   * State colors for interactive elements
   */
  state: {
    /** Disabled elements */
    disabled: palette.gray[600],
    /** Disabled text */
    disabledText: palette.gray[500],
    /** Error state */
    error: '#E53935',
    /** Success state */
    success: '#43A047',
    /** Warning state */
    warning: '#FB8C00',
  },

  /**
   * Border colors
   */
  border: {
    /** Default border */
    default: palette.gray[700],
    /** Subtle border */
    subtle: palette.gray[800],
    /** Accent border */
    accent: palette.brass[500],
  },

  /**
   * Grid and overlay colors
   */
  grid: {
    /** Grid lines */
    line: 'rgba(255, 255, 255, 0.6)',
    /** Grid border */
    border: palette.white,
    /** Handle accent */
    handle: palette.brass[500],
    /** Hole overlay */
    hole: 'rgba(0, 0, 0, 0.7)',
  },
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// Typography Tokens
// ═══════════════════════════════════════════════════════════════════════════

const typography = {
  fontFamily: {
    regular: 'System',
    medium: 'System',
    bold: 'System',
    mono: 'monospace',
  },
  fontSize: {
    xs: 12,
    sm: 13,
    md: 14,
    lg: 16,
    xl: 18,
    '2xl': 20,
    '3xl': 24,
    '4xl': 32,
  },
  fontWeight: {
    regular: '400' as const,
    medium: '500' as const,
    semibold: '600' as const,
    bold: '700' as const,
  },
  lineHeight: {
    tight: 1.2,
    normal: 1.5,
    relaxed: 1.75,
  },
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// Spacing Tokens
// ═══════════════════════════════════════════════════════════════════════════

const spacing = {
  0: 0,
  1: 4,
  2: 8,
  3: 12,
  4: 16,
  5: 20,
  6: 24,
  8: 32,
  10: 40,
  12: 48,
  16: 64,
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// Border Radius Tokens
// ═══════════════════════════════════════════════════════════════════════════

const borderRadius = {
  none: 0,
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  full: 9999,
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// Component Tokens
// ═══════════════════════════════════════════════════════════════════════════

const components = {
  button: {
    /** Standard button height */
    height: 44,
    /** Padding for buttons */
    paddingHorizontal: spacing[6],
    paddingVertical: spacing[3],
    /** Border radius for buttons */
    borderRadius: borderRadius.md,
  },
  control: {
    /** Control button size (row/col increment buttons) */
    size: 36,
    borderRadius: borderRadius.md,
  },
  card: {
    borderRadius: borderRadius.lg,
    padding: spacing[4],
  },
  input: {
    height: 44,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing[3],
  },
} as const;

// ═══════════════════════════════════════════════════════════════════════════
// Export
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Complete design system tokens for the Tactile Game Table theme.
 *
 * Usage:
 * ```typescript
 * import { tokens } from '../theme/tokens';
 *
 * const styles = StyleSheet.create({
 *   button: {
 *     backgroundColor: tokens.colors.accent.primary,
 *     borderRadius: tokens.borderRadius.md,
 *     padding: tokens.spacing[4],
 *   },
 * });
 * ```
 */
export const tokens = {
  colors,
  typography,
  spacing,
  borderRadius,
  components,
} as const;

/** Re-export palette for advanced use cases */
export { palette };

// ═══════════════════════════════════════════════════════════════════════════
// Type Exports
// ═══════════════════════════════════════════════════════════════════════════

export type Tokens = typeof tokens;
export type Colors = typeof colors;
export type AccentColors = typeof colors.accent;
export type SurfaceColors = typeof colors.surface;
export type TextColors = typeof colors.text;
export type StateColors = typeof colors.state;
export type Typography = typeof typography;
export type Spacing = typeof spacing;
export type BorderRadius = typeof borderRadius;
export type Components = typeof components;

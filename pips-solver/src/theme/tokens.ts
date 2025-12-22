/**
 * Design System Tokens
 * "Tactile Game Table" aesthetic - refined luxury meets playful puzzle
 */

// ════════════════════════════════════════════════════════════════════════════
// Color Palette
// ════════════════════════════════════════════════════════════════════════════

export const colors = {
  // Surface colors (dark theme)
  surface: {
    obsidian: '#0D0D0F',    // Deepest background
    charcoal: '#1A1A1F',    // Primary background
    slate: '#2A2A30',       // Card backgrounds
    graphite: '#3D3D45',    // Elevated surfaces
    ash: '#4A4A52',         // Borders, dividers
  },

  // Accent colors (warm metallics)
  accent: {
    brass: '#E8C547',       // Primary accent - buttons, highlights
    brassLight: '#F5D87A',  // Hover state
    brassDark: '#C9A832',   // Pressed state
    copper: '#C17F59',      // Secondary accent
    copperLight: '#D4967A',
  },

  // Semantic colors
  semantic: {
    jade: '#7ECFB3',        // Success, solved, high confidence
    jadeLight: '#A5E0CC',
    jadeDark: '#5FB899',
    coral: '#E85D75',       // Error, low confidence
    coralLight: '#F08090',
    coralDark: '#D04560',
    amber: '#E8C547',       // Warning, medium confidence (same as brass)
  },

  // Region palette (8 refined earth tones)
  regions: [
    '#4A6670',  // Teal Shadow
    '#8B6B5C',  // Warm Stone
    '#5C4A6E',  // Dusty Violet
    '#6B7A4A',  // Olive Drab
    '#6E5A4A',  // Umber
    '#4A5C6E',  // Steel Blue
    '#6E4A5C',  // Mauve
    '#5C6E4A',  // Sage
  ],

  // Domino colors
  domino: {
    ivory: '#F5F0E6',       // Domino tile face
    ivoryDark: '#E8E0D0',   // Domino shadow/edge
    pip: '#2A2A30',         // Pip dots
    border: '#1A1A1F',      // Domino border
  },

  // Text colors
  text: {
    primary: '#FFFFFF',
    secondary: '#A0A0A8',
    tertiary: '#6A6A72',
    inverse: '#0D0D0F',
  },

  // Utility
  transparent: 'transparent',
  overlay: 'rgba(13, 13, 15, 0.85)',
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Typography
// ════════════════════════════════════════════════════════════════════════════

export const typography = {
  fonts: {
    display: 'PlayfairDisplay_700Bold',
    heading: 'DMSans_600SemiBold',
    body: 'DMSans_400Regular',
    bodyMedium: 'DMSans_500Medium',
    mono: 'JetBrainsMono_400Regular',
  },

  // Fallbacks for before fonts load
  fontFallbacks: {
    display: 'Georgia',
    heading: 'System',
    body: 'System',
    mono: 'Courier',
  },

  sizes: {
    xs: 12,
    sm: 14,
    base: 16,
    lg: 18,
    xl: 20,
    '2xl': 24,
    '3xl': 28,
    '4xl': 32,
    '5xl': 40,
  },

  lineHeights: {
    tight: 1.2,
    normal: 1.5,
    relaxed: 1.7,
  },

  letterSpacing: {
    tight: -0.5,
    normal: 0,
    wide: 1,
    extraWide: 2,
  },
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Spacing (8pt grid)
// ════════════════════════════════════════════════════════════════════════════

export const spacing = {
  px: 1,
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
  20: 80,
  24: 96,
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Border Radius
// ════════════════════════════════════════════════════════════════════════════

export const radii = {
  none: 0,
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  '2xl': 24,
  full: 9999,
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Shadows
// ════════════════════════════════════════════════════════════════════════════

export const shadows = {
  none: {
    shadowColor: 'transparent',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0,
    shadowRadius: 0,
    elevation: 0,
  },
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 4,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  glow: {
    shadowColor: colors.accent.brass,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 8,
  },
  innerDomino: {
    // Simulated with gradient overlay in SVG
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 3,
    elevation: 2,
  },
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Animation Timing
// ════════════════════════════════════════════════════════════════════════════

export const animation = {
  duration: {
    instant: 100,
    fast: 150,
    normal: 250,
    slow: 400,
    slower: 600,
  },
  easing: {
    // Standard easing curves
    ease: 'ease',
    easeIn: 'ease-in',
    easeOut: 'ease-out',
    easeInOut: 'ease-in-out',
    // Spring-like for playful interactions
    spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
  },
  stagger: {
    fast: 30,
    normal: 50,
    slow: 80,
  },
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Grid & Layout
// ════════════════════════════════════════════════════════════════════════════

export const grid = {
  cellSize: 56,           // Base cell size for puzzle grid
  cellPadding: 4,         // Inner padding within cells
  gridPadding: 16,        // Padding around entire grid
  dominoBorderWidth: 2,
  pipSize: 8,             // Diameter of pip dots
  pipSpacing: 14,         // Spacing between pips
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Breakpoints (for responsive design)
// ════════════════════════════════════════════════════════════════════════════

export const breakpoints = {
  sm: 375,
  md: 428,
  lg: 768,
  xl: 1024,
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Z-Index Scale
// ════════════════════════════════════════════════════════════════════════════

export const zIndex = {
  base: 0,
  card: 10,
  sticky: 100,
  modal: 1000,
  tooltip: 1100,
  toast: 1200,
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Confidence Thresholds
// ════════════════════════════════════════════════════════════════════════════

export const confidence = {
  high: 0.9,    // >= 90% shows jade
  medium: 0.8,  // >= 80% shows amber
  // < 80% shows coral
} as const;

// ════════════════════════════════════════════════════════════════════════════
// Complete Theme Export
// ════════════════════════════════════════════════════════════════════════════

export const theme = {
  colors,
  typography,
  spacing,
  radii,
  shadows,
  animation,
  grid,
  breakpoints,
  zIndex,
  confidence,
} as const;

export type Theme = typeof theme;
export type Colors = typeof colors;
export type Typography = typeof typography;
